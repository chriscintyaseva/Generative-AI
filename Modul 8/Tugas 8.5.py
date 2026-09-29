from dataclasses import dataclass
import re


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100
) -> list[Chunk]:

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks: list[Chunk] = []
    current = ""
    current_start = 0
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (
            (current + " " + sentence).strip()
            if current
            else sentence
        )

        if len(candidate) > max_chars and current:
            # Save current chunk
            end = char_offset + len(current)

            chunks.append(
                Chunk(
                    doc_id,
                    chunk_idx,
                    current.strip(),
                    char_offset,
                    end
                )
            )

            chunk_idx += 1

            # Start new chunk with overlap
            # from end of previous chunk
            overlap_start = max(
                0,
                len(current) - overlap_chars
            )

            overlap_text = current[overlap_start:]

            current = (
                overlap_text + " " + sentence
            ).strip()

            char_offset = end - len(overlap_text)

        else:
            current = candidate

    # Save final chunk
    if current.strip():
        end = char_offset + len(current)

        chunks.append(
            Chunk(
                doc_id,
                chunk_idx,
                current.strip(),
                char_offset,
                end
            )
        )

    return chunks
chunks = chunk_by_sentences(
    document.strip(),
    doc_id="intro_to_llms",
    max_chars=300,
    overlap_chars=50
)


for c in chunks:
    print(
        f"Chunk {c.chunk_index} "
        f"({c.char_start}-{c.char_end}): "
        f"{c.text[:80]}..."
    )
    print()