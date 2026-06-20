import json
import sys
sys.path.insert(0, '.')

from src.data.processor import Document, PageContent
from src.data.chunking import get_chunker


def load_doc(path: str) -> Document:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    pages = [PageContent(**p) for p in raw['pages']]
    raw['pages'] = pages
    return Document(**raw)


def test_all_strategies():
    doc = load_doc('data/processed/IRS_Circular_E_2024.json')

    print(f"Document : {doc.doc_id}")
    print(f"Pages    : {doc.num_pages}")
    print()

    for strategy in ['naive', 'page_aware', 'hierarchical']:
        chunker = get_chunker(strategy)
        chunks = chunker.chunk(doc)
        page_chunks = [c for c in chunks if c.metadata.get('level') == 'page']
        sub_chunks  = [c for c in chunks if c.metadata.get('level') == 'chunk']

        print(f"{strategy:15s} -> {len(chunks):4d} total chunks", end="")
        if strategy == 'hierarchical':
            print(f"  ({len(page_chunks)} page-level + {len(sub_chunks)} sub-chunks)", end="")
        print()

    print()


def test_naive_loses_page_info():
    doc = load_doc('data/processed/IRS_Circular_E_2024.json')
    chunks = get_chunker('naive').chunk(doc)

    lost = [c for c in chunks if c.page_number == -1]
    print(f"Naive: {len(lost)}/{len(chunks)} chunks have page_number=-1 (lost page info)")
    assert len(lost) == len(chunks), "All naive chunks should have page_number=-1"
    print("PASS: naive chunker correctly loses all page info")
    print()


def test_page_aware_preserves_page_info():
    doc = load_doc('data/processed/IRS_Circular_E_2024.json')
    chunks = get_chunker('page_aware').chunk(doc)

    lost = [c for c in chunks if c.page_number == -1]
    print(f"PageAware: {len(lost)}/{len(chunks)} chunks have unknown page")
    assert len(lost) == 0, "PageAware chunks should all have a valid page_number"
    print("PASS: page_aware chunker preserves all page info")
    print()


def test_hierarchical_parent_child_link():
    doc = load_doc('data/processed/IRS_Circular_E_2024.json')
    chunks = get_chunker('hierarchical').chunk(doc)

    page_chunks = {c.chunk_id: c for c in chunks if c.metadata.get('level') == 'page'}
    sub_chunks  = [c for c in chunks if c.metadata.get('level') == 'chunk']

    broken = [c for c in sub_chunks if c.parent_chunk_id not in page_chunks]
    print(f"Hierarchical: {len(sub_chunks)} sub-chunks, {len(broken)} broken parent links")
    assert len(broken) == 0, "All sub-chunks must have a valid parent_chunk_id"
    print("PASS: all sub-chunks correctly linked to parent page chunks")
    print()

    sample_sub = sub_chunks[0]
    sample_parent = page_chunks[sample_sub.parent_chunk_id]
    print("--- B-tree link example ---")
    print(f"Sub    chunk_id : {sample_sub.chunk_id}")
    print(f"Parent chunk_id : {sample_parent.chunk_id}")
    print(f"Same page?      : {sample_sub.page_number == sample_parent.page_number}")
    print(f"Sub text (50)   : {sample_sub.text[:50]}...")
    print(f"Page text (50)  : {sample_parent.text[:50]}...")


if __name__ == '__main__':
    test_all_strategies()
    test_naive_loses_page_info()
    test_page_aware_preserves_page_info()
    test_hierarchical_parent_child_link()
    print("\nAll tests passed.")
