"""
Module for chart detection and extraction from PDF documents.
"""

import io
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Any


class ChartDetector:
    """
    Detects and extracts charts from PDF documents using various techniques.
    """
    
    def __init__(
        self, 
        min_image_size: int = 100, 
        chart_detection_threshold: float = 0.7,
        force_chart_extraction: bool = True,
        contour_area_threshold: int = 5000,
        trading_chart_detection: bool = True
    ):
        """
        Initialize the ChartDetector with configuration parameters.
        
        Args:
            min_image_size (int): Minimum width/height for an image to be considered a chart
            chart_detection_threshold (float): Confidence threshold for chart detection
            force_chart_extraction (bool): If True, use aggressive chart extraction for trading docs
            contour_area_threshold (int): Minimum area for contour detection (lower = more sensitive)
            trading_chart_detection (bool): Enable specialized trading chart detection
        """
        self.min_image_size = min_image_size
        self.chart_detection_threshold = chart_detection_threshold
        self.force_chart_extraction = force_chart_extraction
        self.contour_area_threshold = contour_area_threshold
        self.trading_chart_detection = trading_chart_detection
    
    def detect_all_charts(self, doc, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect all charts on a page using multiple detection methods.
        
        Args:
            doc: The PyMuPDF document
            page: The PyMuPDF page
            page_num: The page number
            
        Returns:
            List[Dict[str, Any]]: List of detected charts with metadata
        """
        all_charts = []
        
        # Track chart positions to avoid duplicates
        chart_positions = []
        
        # 1. First detect standard embedded images
        embedded_charts = self.detect_embedded_images(doc, page, page_num)
        all_charts.extend(embedded_charts)
        chart_positions.extend([chart.get('position') for chart in embedded_charts])
        
        # 2. Detect vectorized charts using contour detection
        vector_charts = self.detect_vectorized_charts(page, page_num, chart_positions)
        all_charts.extend(vector_charts)
        chart_positions.extend([chart.get('position') for chart in vector_charts])
        
        # 3. Apply trading-specific chart detection
        if self.trading_chart_detection:
            trading_charts = self.detect_trading_specific_charts(page, page_num, chart_positions)
            all_charts.extend(trading_charts)
            chart_positions.extend([chart.get('position') for chart in trading_charts])
        
        # 4. If no charts found and force_chart_extraction is enabled, extract by page division
        if self.force_chart_extraction and not all_charts:
            forced_charts = self.force_extract_by_page_division(page, page_num)
            all_charts.extend(forced_charts)
        
        return all_charts
    
    def detect_embedded_images(self, doc, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract directly embedded images from a PDF page.
        
        Args:
            doc: The PyMuPDF document
            page: The PyMuPDF page
            page_num: The page number
            
        Returns:
            List[Dict[str, Any]]: List of detected embedded charts
        """
        charts = []
        
        # Get images directly embedded in the PDF
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL image for analysis
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Basic size-based filtering
                if (pil_image.width >= self.min_image_size and 
                    pil_image.height >= self.min_image_size):
                    
                    # Get image position on the page
                    img_rect = page.get_image_bbox(xref)
                    
                    chart = {
                        'type': 'embedded',
                        'image_data': image_bytes,
                        'position': (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                    }
                    charts.append(chart)
            except Exception as e:
                print(f"Error processing embedded image {img_index} on page {page_num}: {e}")
                continue
                
        return charts
    
    def detect_vectorized_charts(
        self, 
        page, 
        page_num: int, 
        existing_positions: List[Tuple[float, float, float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Detect vectorized charts using contour detection.
        
        Args:
            page: The PyMuPDF page
            page_num: The page number
            existing_positions: List of positions of already detected charts
            
        Returns:
            List[Dict[str, Any]]: List of detected vectorized charts
        """
        charts = []
        
        try:
            # Render the page to an image
            zoom = 2  # Adjust for better quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to OpenCV format
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding techniques
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Filter by contour area
                area = cv2.contourArea(contour)
                if area > self.contour_area_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip if too small
                    if w < self.min_image_size or h < self.min_image_size:
                        continue
                    
                    # Scale back to PDF coordinates
                    x0, y0 = x/zoom, y/zoom
                    x1, y1 = (x+w)/zoom, (y+h)/zoom
                    
                    # Skip if we already have a chart in this area
                    if self._has_overlap((x0, y0, x1, y1), existing_positions):
                        continue
                    
                    # Crop the region
                    chart_img = cv_img[y:y+h, x:x+w]
                    
                    # Convert to bytes
                    success, buffer = cv2.imencode(".png", chart_img)
                    if success:
                        image_bytes = buffer.tobytes()
                        
                        chart = {
                            'type': 'vector',
                            'image_data': image_bytes,
                            'position': (x0, y0, x1, y1)
                        }
                        charts.append(chart)
        
        except Exception as e:
            print(f"Error in vectorized chart detection on page {page_num}: {e}")
            
        return charts
    
    def detect_trading_specific_charts(
        self, 
        page, 
        page_num: int, 
        existing_positions: List[Tuple[float, float, float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Specialized detection for trading charts which often have specific characteristics.
        
        Args:
            page: The PyMuPDF page
            page_num: The page number
            existing_positions: List of positions of already detected charts
            
        Returns:
            List[Dict[str, Any]]: List of detected trading charts
        """
        charts = []
        
        try:
            # Render the page at high resolution
            zoom = 3
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to OpenCV format
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection to find chart boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Process top contours (limit to prevent false positives)
            max_charts = 3
            charts_found = 0
            
            for contour in contours:
                if charts_found >= max_charts:
                    break
                    
                area = cv2.contourArea(contour)
                if area < 10000:  # Minimum area for a trading chart
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale back to PDF coordinates
                x0, y0 = x/zoom, y/zoom
                x1, y1 = (x+w)/zoom, (y+h)/zoom
                
                # Skip if we already have a chart in this area
                if self._has_overlap((x0, y0, x1, y1), existing_positions):
                    continue
                
                # Crop the region
                chart_img = cv_img[y:y+h, x:x+w]
                
                # Convert to bytes
                success, buffer = cv2.imencode(".png", chart_img)
                if success:
                    image_bytes = buffer.tobytes()
                    
                    chart = {
                        'type': 'trading',
                        'image_data': image_bytes,
                        'position': (x0, y0, x1, y1)
                    }
                    charts.append(chart)
                    charts_found += 1
        
        except Exception as e:
            print(f"Error in trading chart detection on page {page_num}: {e}")
            
        return charts
    
    def force_extract_by_page_division(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        When all else fails, divide the page into regions and extract potential charts.
        Only used when force_chart_extraction is True and no charts were found by other methods.
        
        Args:
            page: The PyMuPDF page
            page_num: The page number
            
        Returns:
            List[Dict[str, Any]]: List of potential charts
        """
        charts = []
        
        try:
            # Render the page
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to OpenCV format
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
            # Get page dimensions
            height, width = cv_img.shape[:2]
            
            # Extract the middle third of the page which often contains charts
            # Skip the top and bottom which often contain text
            top_third = int(height * 0.3)
            bottom_third = int(height * 0.7)
            
            middle_section = cv_img[top_third:bottom_third, :]
            
            # Convert to bytes
            success, buffer = cv2.imencode(".png", middle_section)
            if success:
                image_bytes = buffer.tobytes()
                
                # Scale coordinates back to PDF space
                x0, y0 = 0, top_third/zoom
                x1, y1 = width/zoom, bottom_third/zoom
                
                chart = {
                    'type': 'forced',
                    'image_data': image_bytes,
                    'position': (x0, y0, x1, y1)
                }
                charts.append(chart)
        
        except Exception as e:
            print(f"Error in forced chart extraction on page {page_num}: {e}")
            
        return charts
    
    def _has_overlap(
        self, 
        position: Tuple[float, float, float, float], 
        existing_positions: List[Tuple[float, float, float, float]]
    ) -> bool:
        """
        Check if a position overlaps with any existing positions.
        
        Args:
            position: Position to check (x0, y0, x1, y1)
            existing_positions: List of existing positions
            
        Returns:
            bool: True if there is significant overlap
        """
        x0, y0, x1, y1 = position
        
        for ex_pos in existing_positions:
            ex_x0, ex_y0, ex_x1, ex_y1 = ex_pos
            
            # Check for overlap
            if (ex_x0 < x1 and ex_x1 > x0 and ex_y0 < y1 and ex_y1 > y0):
                # Calculate overlap area
                overlap_width = min(ex_x1, x1) - max(ex_x0, x0)
                overlap_height = min(ex_y1, y1) - max(ex_y0, y0)
                
                if overlap_width > 0 and overlap_height > 0:
                    overlap_area = overlap_width * overlap_height
                    rect_area = (x1 - x0) * (y1 - y0)
                    
                    # If more than 30% overlaps, consider it a duplicate
                    if rect_area > 0 and overlap_area / rect_area > 0.3:
                        return True
        
        return False
