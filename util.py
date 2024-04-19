from typing import List

def chunk_list(list: List, num_chunks: int) -> List:
    size = len(list)
    chunk_size = size // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i*chunk_size
        if i+1 == num_chunks:
            chunks.append(list[start:])
        else:
            end = start + chunk_size
            chunks.append(list[start:end])
    return chunks