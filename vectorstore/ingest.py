"""Destination Guide Ingestion and Chunking.

Reads all destination guide .txt files from data/destination_guide/raw/ and chunks
strictly by section (VISA & ENTRY, BEST TIME TO VISIT, LOCAL CUSTOMS, PACKING TIPS,
SAFETY & HEALTH), tagging each chunk with rich metadata {city, section, country}.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional
from config.settings import settings


@dataclass
class DocumentChunk:
    chunk_id: str
    city: str
    country: str
    section: str
    content: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "city": self.city,
            "country": self.country,
            "section": self.section,
            "content": self.content,
        }

    @property
    def formatted_text(self) -> str:
        """Text used for embedding and context representation."""
        return f"[{self.city.upper()} - {self.section}]\n{self.content}"


KNOWN_SECTIONS = [
    "VISA & ENTRY",
    "BEST TIME TO VISIT",
    "LOCAL CUSTOMS",
    "PACKING TIPS",
    "SAFETY & HEALTH",
]


def parse_destination_file(file_path: Path) -> list[DocumentChunk]:
    """Parse a single destination guide text file into section-based chunks."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    lines = [line.rstrip() for line in text.splitlines()]

    # Extract City & Country from first header line: DESTINATION GUIDE: <CITY, COUNTRY>
    first_line = lines[0] if lines else ""
    city = file_path.stem.replace("_", " ").title()
    country = "Unknown"

    header_match = re.match(r"^DESTINATION\s+GUIDE:\s*([^,]+)(?:,\s*(.+))?", first_line, re.IGNORECASE)
    if header_match:
        city = header_match.group(1).strip()
        if header_match.group(2):
            country = header_match.group(2).strip()

    # Section regex pattern matching known sections
    section_pattern = re.compile(
        r"^(VISA & ENTRY|BEST TIME TO VISIT|LOCAL CUSTOMS|PACKING TIPS|SAFETY & HEALTH)\b",
        re.IGNORECASE,
    )

    chunks: list[DocumentChunk] = []
    current_section: Optional[str] = None
    current_lines: list[str] = []

    def commit_section():
        if current_section and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunk_id = f"{city.lower()}_{current_section.lower().replace(' ', '_').replace('&', 'and')}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        city=city,
                        country=country,
                        section=current_section,
                        content=content,
                    )
                )

    for line in lines[1:]:
        stripped = line.strip()
        match = section_pattern.match(stripped)
        if match:
            # Commit previously accumulated section
            commit_section()
            current_section = match.group(1).upper()
            current_lines = []
        elif current_section is not None:
            if stripped or current_lines:  # keep non-empty or internal spacing
                current_lines.append(stripped)

    # Commit final section
    commit_section()

    return chunks


def load_raw_destination_files(raw_dir: Optional[Path] = None) -> list[DocumentChunk]:
    """Scan raw_dir for all .txt files and extract section chunks.

    Does not hardcode filenames: dynamically ingests any .txt file in the directory.
    """
    directory = raw_dir or settings.RAW_DATA_DIR
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return []

    all_chunks: list[DocumentChunk] = []
    txt_files = sorted(directory.glob("*.txt"))

    for file_path in txt_files:
        chunks = parse_destination_file(file_path)
        all_chunks.extend(chunks)

    return all_chunks
