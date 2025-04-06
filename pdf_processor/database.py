"""
Module for database integration with Supabase.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from supabase import create_client, Client
import openai
from pydantic import BaseModel, Field
from .extractor import Paragraph, Chart
from .enhancer import EnhancedParagraph, EnhancedChart


class VectorDBClient:
    """
    Client for interacting with Supabase vector database.
    """
    
    def __init__(
        self, 
        supabase_url: str,
        supabase_key: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize the Supabase client.
        
        Args:
            supabase_url (str): Supabase URL
            supabase_key (str): Supabase API key
            openai_api_key (str): OpenAI API key for generating embeddings
            embedding_model (str): OpenAI embedding model name
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        
        # Initialize clients
        self.supabase = create_client(supabase_url, supabase_key)
        openai.api_key = openai_api_key
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text using OpenAI.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not text:
            # Return a zero vector for empty text
            # For text-embedding-ada-002, the dimension is 1536
            return [0.0] * 1536
        
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector for failed embedding
            return [0.0] * 1536
    
    def store_note(
        self,
        title: str,
        content: str,
        section: str = "",
        subsection: str = "",
        tags: List[str] = None
    ) -> str:
        """
        Store a trading note in the database.
        
        Args:
            title (str): Title of the note
            content (str): Full content of the note
            section (str): Section category
            subsection (str): Subsection category
            tags (List[str]): List of tags for the note
            
        Returns:
            str: ID of the created note
        """
        if tags is None:
            tags = []
            
        # Generate embedding for the note
        embedding = self.generate_embedding(f"{title} {section} {subsection}")
        
        # Insert the note
        response = self.supabase.table("trading_notes").insert({
            "title": title,
            "content": content,
            "section": section,
            "subsection": subsection,
            "tags": tags,
            "embedding": embedding
        }).execute()
        
        if not response.data:
            raise Exception("Failed to insert note")
            
        return response.data[0]["id"]
    
    def store_chunks(
        self,
        note_id: str,
        chunks: Dict[int, str],
        enhanced_paragraphs: List[EnhancedParagraph]
    ) -> List[str]:
        """
        Store chunks from a note.
        
        Args:
            note_id (str): ID of the parent note
            chunks (Dict[int, str]): Dictionary mapping chunk IDs to chunk text
            enhanced_paragraphs (List[EnhancedParagraph]): Enhanced paragraphs metadata
            
        Returns:
            List[str]: IDs of the created chunks
        """
        chunk_ids = []
        
        # Group enhanced paragraphs by chunk ID
        paragraphs_by_chunk = {}
        for para in enhanced_paragraphs:
            if para.chunk_id is not None:
                if para.chunk_id not in paragraphs_by_chunk:
                    paragraphs_by_chunk[para.chunk_id] = []
                paragraphs_by_chunk[para.chunk_id].append(para)
        
        # Process each chunk
        for chunk_id, chunk_text in chunks.items():
            # Collect trading terms from all paragraphs in this chunk
            all_terms = set()
            chart_references = set()
            
            # Get related paragraphs
            related_paragraphs = paragraphs_by_chunk.get(chunk_id, [])
            
            for para in related_paragraphs:
                all_terms.update(para.trading_terms)
                chart_references.update(para.chart_references)
            
            # Generate embedding for the chunk
            embedding = self.generate_embedding(chunk_text)
            
            # Insert the chunk
            response = self.supabase.table("note_chunks").insert({
                "notes_id": note_id,
                "content": chunk_text,
                "chunk_index": chunk_id,
                "trading_terms": list(all_terms),
                "chart_ids": list(chart_references),
                "embedding": embedding
            }).execute()
            
            if response.data:
                chunk_ids.append(response.data[0]["id"])
            
            # Add a small delay to prevent rate limiting
            time.sleep(0.1)
        
        return chunk_ids
    
    def store_chart(
        self,
        note_id: str,
        chart: Chart,
        enhanced_chart: EnhancedChart
    ) -> str:
        """
        Store a chart image in the database.
        
        Args:
            note_id (str): ID of the parent note
            chart (Chart): Original chart data
            enhanced_chart (EnhancedChart): Enhanced chart metadata
            
        Returns:
            str: ID of the created chart record
        """
        # Upload image to storage
        file_name = f"{note_id}_{enhanced_chart.position_marker}.png"
        storage_path = file_name
        
        try:
            # Upload to Supabase Storage
            storage_response = self.supabase.storage.from_("chart-images").upload(
                path=storage_path,
                file=chart.image_data,
                file_options={"content-type": "image/png"}
            )
            
            if hasattr(storage_response, 'error') and storage_response.error:
                raise Exception(f"Error uploading image: {storage_response.error}")
            
            # Generate embedding for the chart based on caption and tags
            embedding_text = f"""
            Caption: {enhanced_chart.caption or ''}
            Tags: {', '.join(enhanced_chart.tags or [])}
            """
            
            embedding = self.generate_embedding(embedding_text)
            
            # Create chart record
            img_insert = self.supabase.table("chart_images").insert({
                "file_path": storage_path,
                "caption": enhanced_chart.caption,
                "notes_id": note_id,
                "position_marker": enhanced_chart.position_marker,
                "tags": enhanced_chart.tags,
                "embedding": embedding
            }).execute()
            
            if hasattr(img_insert, 'error') and img_insert.error:
                raise Exception(f"Error inserting image record: {img_insert.error}")
            
            return img_insert.data[0]["id"]
            
        except Exception as e:
            print(f"Error storing chart: {e}")
            return None
    
    def find_similar_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a query using vector search.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Similar chunks
        """
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Perform vector search
        # Supabase uses pgvector for vector similarity search
        distance_type = "cosine"  # or "euclidean" or "dot_product"
        
        # Using PostgreSQL function for vector similarity search
        rpc_response = self.supabase.rpc(
            "search_chunks",
            {
                "query_embedding": query_embedding,
                "similarity_threshold": 0.7,
                "match_count": limit
            }
        ).execute()
        
        return rpc_response.data
    
    def find_similar_charts(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find charts similar to a query using vector search.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Similar charts
        """
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Perform vector search on chart descriptions
        rpc_response = self.supabase.rpc(
            "search_charts",
            {
                "query_embedding": query_embedding,
                "similarity_threshold": 0.7,
                "match_count": limit
            }
        ).execute()
        
        return rpc_response.data
    
    def get_chart_by_id(self, chart_id: str) -> Dict[str, Any]:
        """
        Get a chart by its ID.
        
        Args:
            chart_id (str): ID of the chart
            
        Returns:
            Dict[str, Any]: Chart data
        """
        response = self.supabase.table("chart_images").select("*").eq("id", chart_id).execute()
        
        if not response.data:
            return None
            
        return response.data[0]
    
    def get_related_charts_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get charts related to a specific chunk.
        
        Args:
            chunk_id (str): ID of the chunk
            
        Returns:
            List[Dict[str, Any]]: Related charts
        """
        # Get the chunk to find chart references
        chunk_response = self.supabase.table("note_chunks").select("*").eq("id", chunk_id).execute()
        
        if not chunk_response.data:
            return []
            
        chart_ids = chunk_response.data[0].get("chart_ids", [])
        
        if not chart_ids:
            return []
            
        # Get the charts
        charts = []
        for chart_id in chart_ids:
            chart = self.get_chart_by_id(chart_id)
            if chart:
                charts.append(chart)
                
        return charts
