from typing import List, Optional
from haystack.dataclasses import ByteStream
from fastapi import UploadFile
from hayhooks import BasePipelineWrapper, log
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup a pipeline with a default Qdrant collection"""
        # Create a reusable document store
        self.document_store = QdrantDocumentStore(
            host="qdrant",
            index="default",
            recreate_index=False,  # Do not overwrite existing collection
            embedding_dim=768
        )

        # Create a single pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("converter", TextFileToDocument())
        self.pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        self.pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        self.pipeline.connect("converter", "embedder")
        self.pipeline.connect("embedder", "writer")

    def run_api(self, files: Optional[List[UploadFile]] = None, collection_name: str = "default") -> dict:
        if not files:
            log.debug("No files to index")
            return {"success": True}

        # Update document store to use the requested collection
        if self.document_store.index != collection_name:
            self.document_store = QdrantDocumentStore(
                host="qdrant",
                index=collection_name,
                recreate_index=False,  # Do not overwrite existing data
                embedding_dim=768
            )
            # Update writer in the pipeline to point to the new collection
            self.pipeline.remove_component("writer")
            self.pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
            self.pipeline.connect("embedder", "writer")

        # Prepare all files for a single pipeline run
        sources = []
        for file in files:
            text = file.file.read().decode("utf-8")
            sources.append(ByteStream(text.encode()))
            log.debug(f"Prepared file: {file.filename} for indexing")

        # Run pipeline once for all files
        self.pipeline.run({"converter": {"sources": sources}})

        return {"success": True}
