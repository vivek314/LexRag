import re
from dataclasses import dataclass, field
from typing import Optional
from src.data.processor import Document, PageContent
from abc import ABC, abstractmethod

@dataclass
class Chunk:
    chunk_id: str          # "{doc_id}_p{page}_c{chunk_index}"
    doc_id: str
    text: str
    page_number: int       # Which page this chunk came from
    chunk_index: int       # Position within the document
    char_count: int
    parent_chunk_id: Optional[str] = None   # For hierarchical: links chunk -> page
    references: list = field(default_factory=list)   # Section IDs referenced in text e.g. ["43", "2"]
    section_id: Optional[str] = None                 # "43", "3A" etc. for statute chunks
    metadata: dict = field(default_factory=dict)

class ChunkingStrategy(ABC):

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod    
    def chunk(self, doc: Document) -> list[Chunk]:
        pass

class NaiveChunker(ChunkingStrategy):
    def chunk(self, doc: Document) -> list[Chunk]:
        # Flatten ALL pages into one big string — ignore page boundaries
        full_text = "\n".join(p.text for p in doc.pages)

        chunks = []
        start = 0
        index = 0

        while start < len(full_text):
            end = start + self.chunk_size
            text = full_text[start:end].strip()

            if len(text) > 0:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_c{index}",
                    doc_id=doc.doc_id,
                    text=text,
                    page_number=-1,       # -1 = unknown, we lost page info!
                    chunk_index=index,
                    char_count=len(text),
                ))
                index += 1

            start += self.chunk_size - self.overlap  # slide forward with overlap

        return chunks

class PageAwareChunker(ChunkingStrategy):
    def chunk(self, doc: Document) -> list[Chunk]:
        chunks = []
        index = 0

        for page in doc.pages:
            start = 0
            text = page.text

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if len(chunk_text) > 0:
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_p{page.page_number}_c{index}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        page_number=page.page_number,   # we KNOW which page!
                        chunk_index=index,
                        char_count=len(chunk_text),
                    ))
                    index += 1

                start += self.chunk_size - self.overlap

        return chunks

class HierarchicalChunker(ChunkingStrategy):
    def chunk(self, doc: Document) -> list[Chunk]:
        all_chunks = []
        index = 0

        for page in doc.pages:

            # --- Pass 1: Page-level chunk (index node) ---
            page_chunk_id = f"{doc.doc_id}_page_{page.page_number}"
            all_chunks.append(Chunk(
                chunk_id=page_chunk_id,
                doc_id=doc.doc_id,
                text=page.text,
                page_number=page.page_number,
                chunk_index=index,
                char_count=len(page.text),
                parent_chunk_id=None,
                metadata={"level": "page"}
            ))
            index += 1

            # --- Pass 2: Sub-chunks within this page (leaf nodes) ---
            start = 0
            while start < len(page.text):
                end = start + self.chunk_size
                chunk_text = page.text[start:end].strip()

                if len(chunk_text) > 0:
                    all_chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_page_{page.page_number}_c{index}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        page_number=page.page_number,
                        chunk_index=index,
                        char_count=len(chunk_text),
                        parent_chunk_id=page_chunk_id,  # foreign key to page chunk
                        metadata={"level": "chunk"}
                    ))
                    index += 1

                start += self.chunk_size - self.overlap

        return all_chunks

class StatuteChunker(ChunkingStrategy):
    """
    Splits legal statute PDFs by Section -> Sub-section/Clause boundaries.

    Hierarchy:
      level="section"  — one chunk per statute section ("43. Penalty for damage...")
                         This is the reference target when another section says "section 43".
      level="clause"   — sub-chunks within a section (what the LLM actually reads).
                         parent_chunk_id points back to the section chunk.

    Both levels carry:
      section_id  : "43" or "3A" — used for O(1) lookup during reference resolution
      references  : list of section IDs mentioned inside this chunk's text
    """

    # Matches statute section starts: "43. Penalty..." or "3A. Electronic..."
    # Also handles Indian statute PDFs that insert footnote numbers before the title:
    #   "43.  6 [Penalty and compensation] for damage..."
    # Pattern breakdown:
    #   (\d+[A-Z]?)  — section number (43, 3A, 66F …)
    #   \.\s+        — period + whitespace
    #   (?:\d+\s+)?  — optional footnote number (e.g. "6 ")
    #   (?:\[)?      — optional opening bracket (e.g. "[")
    #   [A-Z][a-z]   — title word must start uppercase (filters out sub-section bodies)
    SECTION_RE = re.compile(
        r'^(\d+[A-Z]?)\.\s+(?:\d+\s+)?(?:\[)?[A-Z][a-z]',
        re.MULTILINE,
    )

    # Minimum body length: shorter texts are Table-of-Contents entries or stubs.
    _MIN_SECTION_CHARS = 150

    # Amendment annotations (not actual section bodies).
    # Catches two forms:
    #   1. Verb-prefix: "7. Ins. by ...", "43. Subs. by ...", "7. Clauses (a) omitted by ..."
    #   2. Body-level: any text whose first 120 chars contain "by Act N of YYYY" or "by s. N"
    #      e.g. "7. Clauses (r), (s) and (t) omitted by Act 7 of 2017, s. 169..."
    _AMENDMENT_RE = re.compile(r'^\d+[A-Z]?\.\s+(?:Ins|Subs|Omit|Sub|Clauses?)\b', re.IGNORECASE)
    _AMENDMENT_BODY_RE = re.compile(r'\bby\s+(?:Act\s+\d+\s+of\s+\d{4}|s\.\s*\d+)', re.IGNORECASE)

    # Type 3 (most specific first): clause (a) of sub-section (2) of section 6
    _CLAUSE_REF_RE = re.compile(
        r'\bclause\s+\([a-z]\)\s+of\s+sub-section\s+\(\d+\)\s+of\s+section\s+(\d+[A-Z]?)\b',
        re.IGNORECASE,
    )
    # Type 2: sub-section (1) of section 46
    _SUBSEC_REF_RE = re.compile(
        r'\bsub-section\s+\(\d+\)\s+of\s+section\s+(\d+[A-Z]?)\b',
        re.IGNORECASE,
    )
    # Type 1: section 3
    _SEC_REF_RE = re.compile(r'\bsection\s+(\d+[A-Z]?)\b', re.IGNORECASE)

    def chunk(self, doc: Document) -> list[Chunk]:
        # --- Step 1: Join all pages into one string, track page-start positions ---
        page_starts: list[tuple[int, int]] = []   # (char_offset, page_number)
        full_text = ""
        for page in doc.pages:
            page_starts.append((len(full_text), page.page_number))
            full_text += page.text + "\n"

        # --- Step 2: Find all section boundaries ---
        section_matches = list(self.SECTION_RE.finditer(full_text))

        if not section_matches:
            # Not a statute (no section markers found) — fall back to page-aware chunking
            return PageAwareChunker(self.chunk_size, self.overlap).chunk(doc)

        # --- Step 3: Slice text into sections, deduplicate by section_id ---
        # The Table of Contents produces short duplicate matches (e.g. "1. Short title")
        # alongside the real section body. Keep only the longest text per section_id.
        raw_sections: list[tuple[str, int, str]] = []   # (section_id, char_pos, text)
        for i, match in enumerate(section_matches):
            section_id = match.group(1)
            sec_start  = match.start()
            sec_end    = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(full_text)
            )
            section_text = full_text[sec_start:sec_end].strip()
            if section_text:
                raw_sections.append((section_id, sec_start, section_text))

        # Deduplicate: for each section_id keep the best candidate.
        # Priority: longest non-amendment text. If only a short ToC entry exists
        # (e.g. section body uses a mid-line format the regex can't reach), keep
        # that entry — it's better than nothing for embedding search.
        best: dict[str, tuple[int, str]] = {}        # section_id -> (char_pos, text)
        best_is_stub: dict[str, bool]    = {}        # track whether current best is a stub

        for sec_id, pos, text in raw_sections:
            # Skip amendment annotations (e.g. "7. Ins. by s. 21" or "7. Clauses omitted by Act 7 of 2017")
            if self._AMENDMENT_RE.match(text):
                continue
            if self._AMENDMENT_BODY_RE.search(text[:120]):
                continue
            is_stub = len(text) < self._MIN_SECTION_CHARS
            if sec_id not in best:
                best[sec_id] = (pos, text)
                best_is_stub[sec_id] = is_stub
            else:
                current_is_stub = best_is_stub[sec_id]
                # Prefer body text over ToC stub; among same kind, keep longest
                if current_is_stub and not is_stub:
                    best[sec_id] = (pos, text)
                    best_is_stub[sec_id] = False
                elif current_is_stub == is_stub and len(text) > len(best[sec_id][1]):
                    best[sec_id] = (pos, text)

        # Build chunks from deduplicated sections, sorted by appearance order
        all_chunks: list[Chunk] = []
        index = 0

        for section_id, (sec_start, section_text) in sorted(best.items(),
                                                              key=lambda x: x[1][0]):
            start_page = self._char_to_page(sec_start, page_starts)

            # Extract cross-references (all three forms), remove self-reference
            references = self._extract_references(section_text, section_id)

            # --- Level 1: Section chunk (index node / reference target) ---
            section_chunk_id = f"{doc.doc_id}_sec_{section_id}"
            all_chunks.append(Chunk(
                chunk_id=section_chunk_id,
                doc_id=doc.doc_id,
                text=section_text,
                page_number=start_page,
                chunk_index=index,
                char_count=len(section_text),
                parent_chunk_id=None,
                references=references,
                section_id=section_id,
                metadata={"level": "section"},
            ))
            index += 1

            # --- Level 2: Clause chunks (leaf nodes — what LLM reads) ---
            start = 0
            while start < len(section_text):
                end = start + self.chunk_size
                chunk_text = section_text[start:end].strip()
                if chunk_text:
                    all_chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_sec_{section_id}_c{index}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        page_number=start_page,
                        chunk_index=index,
                        char_count=len(chunk_text),
                        parent_chunk_id=section_chunk_id,
                        references=references,   # inherit — clauses may cite same sections
                        section_id=section_id,
                        metadata={"level": "clause"},
                    ))
                    index += 1
                start += self.chunk_size - self.overlap

        return all_chunks

    def _extract_references(self, text: str, own_section_id: str) -> list[str]:
        """
        Extract all referenced section IDs from text, covering three forms:
          Type 3: clause (a) of sub-section (2) of section 6  -> "6"
          Type 2: sub-section (1) of section 46               -> "46"
          Type 1: section 3                                    -> "3"

        Apply most-specific pattern first; use a set so "section 46" appearing
        inside a Type-2 phrase doesn't double-count. Self-references are removed.
        """
        found: set[str] = set()
        # Type 3 — clause ref (section number is the only capture group)
        for m in self._CLAUSE_REF_RE.finditer(text):
            found.add(m.group(1))
        # Type 2 — sub-section ref
        for m in self._SUBSEC_REF_RE.finditer(text):
            found.add(m.group(1))
        # Type 1 — plain section ref (catches remaining simple mentions)
        for m in self._SEC_REF_RE.finditer(text):
            found.add(m.group(1))
        found.discard(own_section_id)
        return list(found)

    def _char_to_page(self, pos: int, page_starts: list[tuple[int, int]]) -> int:
        """Return the page number that contains character position `pos`."""
        for start, pnum in reversed(page_starts):
            if pos >= start:
                return pnum
        return page_starts[0][1]


def get_chunker(strategy: str, chunk_size: int = 512, overlap: int = 50) -> ChunkingStrategy:
    strategies = {
        "naive": NaiveChunker(chunk_size, overlap),
        "page_aware": PageAwareChunker(chunk_size, overlap),
        "hierarchical": HierarchicalChunker(chunk_size, overlap),
        "statute": StatuteChunker(chunk_size, overlap),
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    return strategies[strategy]