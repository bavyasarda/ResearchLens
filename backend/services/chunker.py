"""
Core Chunker Service - Context-aware text chunking for paper abstracts.
Implements sentence-aware splitting with overlapping context windows.
"""
import logging
import re
from typing import List, Optional

from backend.models.schemas import Chunk, Paper

logger = logging.getLogger(__name__)


class CoreChunker:
    """
    Core Chunk Technique for document chunking:

    - Sentence-aware splitting (never cut mid-sentence)
    - Each chunk carries: text, paper_id, section_type, position, context_window
    - Context window = previous chunk's last sentence + current chunk + next chunk's first sentence
    - Overlapping windows for boundary preservation

    This improves embedding quality without inflating context size at generation time.
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        """
        Initialize the Core Chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_paper(self, paper: Paper) -> List[Chunk]:
        """
        Chunk the paper's abstract into context-aware segments.

        Args:
            paper: Paper object with abstract and metadata

        Returns:
            List of Chunk objects with text, context_text, and metadata
        """
        abstract = paper.abstract or ""
        if not abstract.strip():
            # Return a single chunk with title as fallback
            return [self._create_chunk(paper, paper.title, 0)]

        # Split into sentences
        sentences = self._split_into_sentences(abstract)

        if not sentences:
            return [self._create_chunk(paper, abstract, 0)]

        chunks = []
        current_chunk_text = ""
        position = 0

        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if current_chunk_text and (
                len(current_chunk_text) + len(sentence) > self.chunk_size
                and current_chunk_text.strip()
            ):
                # Save current chunk
                chunks.append(self._create_chunk(paper, current_chunk_text.strip(), position))
                position += 1

                # Start new chunk with overlap from previous
                overlap_text = self._get_overlap_text(current_chunk_text, self.overlap)
                current_chunk_text = overlap_text

            # Add sentence to current chunk
            if current_chunk_text:
                current_chunk_text += " " + sentence
            else:
                current_chunk_text = sentence

        # Don't forget the last chunk
        if current_chunk_text.strip():
            chunks.append(self._create_chunk(paper, current_chunk_text.strip(), position))

        return chunks if chunks else [self._create_chunk(paper, abstract, 0)]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving sentence boundaries.
        Handles common abbreviations and decimal numbers.

        Args:
            text: Input text to split

        Returns:
            List of sentences
        """
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split on sentence boundaries (., !, ?) but not abbreviations
        # Use a lookahead to not split on common abbreviations
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.',
                       'et al.', 'e.g.', 'i.e.', 'vs.', 'fig.', 'Eq.', 'Ref.']

        # Temporarily replace abbreviations
        temp_text = text
        abbrev_replacements = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            temp_text = temp_text.replace(abbr, placeholder)
            abbrev_replacements[placeholder] = abbr

        # Split on sentence-ending punctuation followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', temp_text)

        # Restore abbreviations
        sentences = [s for s in sentences if s.strip()]
        return sentences

    def _create_chunk(self, paper: Paper, text: str, position: int) -> Chunk:
        """
        Create a Chunk with context window.

        Args:
            paper: Parent paper
            text: Raw chunk text
            position: Position in document

        Returns:
            Chunk with context_text for embedding
        """
        # Metadata for the chunk
        metadata = {
            "year": paper.year,
            "authors": paper.authors,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "source": paper.source,
        }

        chunk_id = f"{paper.paper_id}_chunk_{position}"

        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            text=text,
            context_text=text,  # Can be enriched with window in build_context_window
            position=position,
            metadata=metadata,
        )

    def _get_overlap_text(self, text: str, max_chars: int) -> str:
        """
        Get overlapping text from the end of the current chunk.
        Takes the last sentence(s) up to max_chars.

        Args:
            text: Current chunk text
            max_chars: Maximum overlap characters

        Returns:
            Overlap text
        """
        if len(text) <= max_chars:
            return text

        # Find sentence boundary near max_chars
        cutoff = text[-max_chars:]
        match = re.search(r'\.\s+', cutoff)
        if match:
            return cutoff[match.end():]

        # If no sentence boundary, take last portion
        return cutoff.strip()

    def build_context_window(self, chunks: List[Chunk], idx: int) -> str:
        """
        Build a context-enriched text for a chunk at given index.

        The context window includes:
        - Previous chunk's last sentence (if available)
        - Current chunk text
        - Next chunk's first sentence (if available)

        Args:
            chunks: List of all chunks from the document
            idx: Index of the current chunk

        Returns:
            Context-enriched text for embedding
        """
        if not chunks or idx >= len(chunks):
            return ""

        current_text = chunks[idx].text

        # Get previous chunk's last sentence
        prev_context = ""
        if idx > 0:
            prev_text = chunks[idx - 1].text
            sentences = self._split_into_sentences(prev_text)
            if sentences:
                prev_context = sentences[-1] + " "

        # Get next chunk's first sentence
        next_context = ""
        if idx < len(chunks) - 1:
            next_text = chunks[idx + 1].text
            sentences = self._split_into_sentences(next_text)
            if sentences:
                next_context = " " + sentences[0]

        return prev_context + current_text + next_context

    def chunk_all_papers(self, papers: List[Paper]) -> List[Chunk]:
        """
        Chunk all papers and return a flat list of chunks.

        Args:
            papers: List of Paper objects

        Returns:
            Flat list of all chunks from all papers
        """
        all_chunks = []
        for paper in papers:
            paper_chunks = self.chunk_paper(paper)
            # Update context_text with window
            for i, chunk in enumerate(paper_chunks):
                chunk.context_text = self.build_context_window(paper_chunks, i)
            all_chunks.extend(paper_chunks)

        logger.info(f"Chunked {len(papers)} papers into {len(all_chunks)} chunks")
        return all_chunks