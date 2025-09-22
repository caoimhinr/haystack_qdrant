import os
import tempfile
from typing import Optional
from git import Repo
from haystack.dataclasses import Document, ByteStream
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from hayhooks import BasePipelineWrapper, log

class GitPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup a dummy pipeline so hayhooks can register the wrapper."""
        document_store = QdrantDocumentStore(
            host="qdrant",
            index="default",
            embedding_dim=768
        )
        pipeline = Pipeline()
        pipeline.add_component("converter", TextFileToDocument())
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        pipeline.connect("converter", "embedder")
        pipeline.connect("embedder", "writer")
        self.pipeline = pipeline

    def run_api(
        self,
        git_url: str,
        pat: str,
        collection_name: str = "default"
    ) -> dict:
        """Clone a Git repo and index all files with filenames as metadata."""
        # Prepare Qdrant document store
        document_store = QdrantDocumentStore(
            host="qdrant",
            index=collection_name,
            embedding_dim=768,
            recreate_index=False  # Set True if you want to overwrite
        )

        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component("converter", TextFileToDocument())
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        pipeline.connect("converter", "embedder")
        pipeline.connect("embedder", "writer")

        # Clone repo to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_url_with_auth = git_url.replace(
                "https://", f"https://{pat}@"
            )
            log.debug(f"Cloning {git_url} into {tmpdir}")
            Repo.clone_from(repo_url_with_auth, tmpdir)

            # Walk through all files
            for root, _, files in os.walk(tmpdir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        log.debug(f"Indexing file: {filename}")
                        doc = Document(
                            content=content,
                            meta={"filename": filename, "filepath": os.path.relpath(file_path, tmpdir)}
                        )
                        pipeline.run({"converter": {"sources": [ByteStream(content.encode())], "meta": doc.meta}})
                    except Exception as e:
                        log.warning(f"Skipping file {filename}: {e}")

        return {"success": True}
