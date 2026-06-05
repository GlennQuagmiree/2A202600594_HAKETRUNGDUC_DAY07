# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Hà Kế Trung Đức]
**Nhóm:** [A6]
**Ngày:** [5/6/2025]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Khi góc giữa 2 vecto A và B càng nhỏ thì 2 vecto này càng cùng chỉ về 1 hướng,có nghĩa là cosine similarity càng tiệm cận về 1 chứng tỏ là vecto của 2 câu này càng có độ tương đồng cao.Bất kể 2 vecto của 2 câu này có chiều dài bao nhiêu, nếu 2 câu này có nội dung tương đồng với nhau thì cosine similarity sẽ cao.

**Ví dụ HIGH similarity:**
- Sentence A: "Xin chào, tôi là học sinh của trường THPT Hà Nội"
- Sentence B: "Chào bạn, tôi là học sinh của trường THPT Hà Nội"
- Tại sao tương đồng: hai câu này có nội dung tương đồng với nhau, chỉ khác nhau một vài từ không quan trọng đến ngữ nghĩa

**Ví dụ LOW similarity:**
- Sentence A: "Tôi đang học lập trình"
- Sentence B: "Thời tiết hôm nay rất đẹp"
- Tại sao khác: hai câu này có nội dung hoàn toàn khác nhau, không có từ nào chung

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:Bởi vì Euclidean distance cho text embedding sẽ bị ảnh hưởng bởi độ dài của câu, trong khi cosine similarity thì không.Nếu hai câu có độ tương đồng cao nhưng độ dài khác nhau thì Euclidean distance sẽ cho kết quả sai lệch, trong khi cosine similarity sẽ cho kết quả đúng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* 

( 10000-50)/(500-50)=22.11
Do 450*0.11=49.5 mà 49.5<50 thì ta phải thêm 1 chunk nữa để chứa phần thừa

> *Đáp án:23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> - **Thay đổi về số lượng chunk:** Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên thành **25 chunks** (áp dụng công thức: `ceil((10000 - 100) / (500 - 100)) = ceil(24.75) = 25`). Điều này xảy ra do bước di chuyển (step size = chunk_size - overlap) giảm từ 450 xuống còn 400.
> - **Tại sao muốn overlap nhiều hơn:** Giúp hạn chế tình trạng một câu hoặc một ý nghĩa quan trọng bị cắt làm đôi ngay tại ranh giới giữa hai chunk, bảo toàn ngữ cảnh liên tục giữa các chunk kế cận và tăng hiệu quả truy xuất thông tin (retrieval) chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI & Data Engineering — Hệ thống RAG và Truy xuất Tri thức Nội bộ

**Tại sao nhóm chọn domain này?**
> Domain này trực tiếp liên quan đến nội dung đang học trong Lab 7. Các tài liệu về Python, RAG System Design, và Vector Store giúp chúng ta vừa xây dựng hệ thống vừa kiểm thử trên nội dung có độ phức tạp thực tế. Domain AI Engineering cũng có cấu trúc rõ ràng (khái niệm → kiến trúc → vận hành) rất phù hợp để đánh giá các chiến lược chunking và retrieval.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | Viết thủ công (AI Engineering) | 1944 | source, language=english, type=tutorial |
| 2 | rag_system_design.md | Viết thủ công (RAG Architecture) | 2391 | source, language=english, type=design |
| 3 | vector_store_notes.md | Viết thủ công (Vector DB concepts) | 2123 | source, language=english, type=tutorial |
| 4 | vi_retrieval_notes.md | Viết thủ công (Vietnamese retrieval notes) | 1667 | source, language=vietnamese, type=tutorial |
| 5 | customer_support_playbook.txt | Viết thủ công (Support operations) | 1692 | source, language=english, type=design |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | python_intro.txt | Trace back câu trả lời về tài liệu gốc |
| language | string | english / vietnamese | Lọc theo ngôn ngữ khi câu hỏi của người dùng bằng tiếng Việt |
| type | string | tutorial / design | Phân loại tài liệu hướng dẫn vs thiết kế để tránh trả về sai loại |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (chunk_size=200, overlap=20) | 11 | 194.9 | Có (overlap giữ liên kết) |
| python_intro.txt | SentenceChunker (max_sentences=3) | 5 | 387.0 | Tốt nhất (giữ trọn câu) |
| python_intro.txt | RecursiveChunker (chunk_size=200) | 14 | 136.9 | Trung bình (chunk khá nhỏ) |
| rag_system_design.md | FixedSizeChunker | 14 | 189.4 | Có |
| rag_system_design.md | SentenceChunker | 5 | 476.0 | Tốt (nhưng chunk dài hơn) |
| rag_system_design.md | RecursiveChunker | 20 | 117.7 | Trung bình |
| vector_store_notes.md | FixedSizeChunker | 12 | 195.2 | Có |
| vector_store_notes.md | SentenceChunker | 8 | 263.6 | Tốt |
| vector_store_notes.md | RecursiveChunker | 18 | 116.1 | Trung bình |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=300)

**Mô tả cách hoạt động:**
> RecursiveChunker ưu tiên tách văn bản theo thứ tự: `\n\n` (đoạn văn) → `\n` (dòng) → `. ` (câu) → ` ` (từ). Nó sẽ cố gắng tách ở mức cao nhất trước, chỉ dùng mức chi tiết hơn nếu chunk vẫn còn quá lớn. Với `chunk_size=300`, mỗi chunk có kích thước cân bằng hơn so với baseline 200, đủ để chứa nhiều hơn một ý trong mỗi chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu về AI Engineering thường có cấu trúc theo đoạn văn rõ ràng (mỗi đoạn trình bày một ý). RecursiveChunker tôn trọng ranh giới đoạn văn này, giúp mỗi chunk mang trọn một ý hoàn chỉnh. Chunk 300 ký tự là điểm cân bằng tốt: đủ ngắn để retrieval chính xác, đủ dài để giữ ngữ cảnh.

**Code snippet (nếu custom):**
```python
# Không custom, dùng RecursiveChunker với chunk_size tùy chỉnh
chunker = RecursiveChunker(chunk_size=300)
chunks = chunker.chunk(document_text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| rag_system_design.md | RecursiveChunker(200) — baseline | 20 | 117.7 | Thấp hơn (chunk quá nhỏ, mất ngữ cảnh) |
| rag_system_design.md | **RecursiveChunker(300) — của tôi** | **15** | **157.5** | **Cao hơn (cân bằng hơn)** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Trung Đức) | RecursiveChunker(300) | 7/10 | Tôn trọng cấu trúc tài liệu, chunk cân bằng | Không có overlap nên một số ý vẫn bị cắt |
| [Thành viên khác] | SentenceChunker | 8/10 | Bảo toàn câu hoàn chỉnh, dễ đọc | Chunk dài hơn (476 ký tự), khó khớp query ngắn |
| [Thành viên khác] | FixedSizeChunker | 6/10 | Đơn giản, dễ kiểm soát | Cắt giữa câu, mất ngữ cảnh |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> SentenceChunker phù hợp nhất với tài liệu kỹ thuật có cấu trúc câu rõ ràng vì nó bảo toàn câu hoàn chỉnh. RecursiveChunker với chunk_size=300 là lựa chọn thực dụng vì dễ kiểm soát kích thước và tôn trọng cấu trúc đoạn văn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=\. |\! |\? |\.\n)` để phát hiện kết thúc câu bằng lookbehind assertion — không xóa dấu chấm câu trong kết quả. Sau đó gom từng nhóm `max_sentences_per_chunk` câu lại thành một chunk bằng `join`. Edge case được xử lý: text rỗng trả về `[]`, các câu trống được lọc bằng `strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy: thử tách bằng `separator[0]` trước. Nếu một phần sau khi tách vẫn còn dài hơn `chunk_size`, gọi đệ quy `_split(part, remaining_separators[1:])` để dùng separator nhỏ hơn. Base case là khi `len(current_text) <= chunk_size` hoặc không còn separator nào. Kết quả các chunks được gom lại theo greedy strategy: thêm part vào chunk hiện tại cho đến khi vượt quá `chunk_size` thì flush.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` duyệt từng Document, gọi `embedding_fn(doc.content)` để tạo vector, rồi lưu dict `{id, content, metadata, embedding}` vào `self._store` (list). Khi có ChromaDB, dùng `collection.add()` để lưu. `search` embed query rồi tính cosine similarity với mọi stored embedding, sort descending và trả về top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter **trước** — lọc `self._store` theo metadata_filter (kiểm tra từng key-value) để tạo `filtered_records`, rồi chạy similarity search chỉ trên tập đã lọc. `delete_document` so sánh `r["id"] == doc_id` để loại bỏ tất cả records có ID khớp, trả về `True` nếu đã xóa ít nhất 1 record.

### KnowledgeBaseAgent

**`answer`** — approach:
> `answer` gọi `self.store.search(question, top_k)` để lấy top-k chunks. Nối nội dung các chunks bằng `\n\n`. Xây dựng prompt theo cấu trúc: `Context information is below. ... Given the context information and not prior knowledge, answer the query. Query: ... Answer:`. Cuối cùng gọi `self.llm_fn(prompt)` để LLM sinh câu trả lời bám sát ngữ cảnh.

### Test Results

```
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.08s ==============================
```

**Số tests pass:** 42 / 42


---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score (Mock / Local) | Đúng? |
|------|-----------|-----------|---------|-----------------------------|-------|
| 1 | The weather today is very nice and sunny. | The weather today is very nice and sunny. | High | 1.0000 / 1.0000 | Đúng |
| 2 | The dog chased the cat up the tree. | A canine ran after the feline up the oak. | High | 0.0466 / 0.5651 | Đúng (với Local) |
| 3 | I love coding in Python and building AI models. | Software engineering is a great career path. | Medium | 0.1397 / 0.4792 | Đúng (với Local) |
| 4 | We are learning about data foundations today. | Apples are red and grow on trees. | Low | 0.1509 / 0.0758 | Đúng |
| 5 | The room is extremely hot. | The room is freezing cold. | Low | 0.1191 / 0.5992 | Sai (với Local) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> - **Kết quả bất ngờ nhất:** Cặp 5 ("extremely hot" và "freezing cold") có điểm tương đồng rất cao (0.5992 với Local Embedder) mặc dù mang ý nghĩa hoàn toàn ngược nhau về mặt logic. Đồng thời, Mock Embedder (dựa trên MD5) cho điểm cực kỳ thấp đối với các từ đồng nghĩa (Cặp 2: 0.0466).
> - **Bài học rút ra:** Điều này chứng minh rằng các mô hình embedding biểu diễn nghĩa dựa trên **ngữ cảnh phân bố (distributional context)**. Do hai câu ngược nghĩa có cấu trúc giống hệt nhau và các từ "hot", "cold" thường xuất hiện trong các văn cảnh tương tự nhau, vector của chúng vẫn nằm rất gần nhau trong không gian. Cosine similarity phản ánh độ tương quan chủ đề/ngữ cảnh chứ không phản ánh tính phủ định hay logic. Đối với Mock Embedder, vì chỉ băm ký tự đơn thuần nên nó hoàn toàn thất bại trong việc nhận diện các từ đồng nghĩa.


---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is Python and what is it used for? | Python là ngôn ngữ lập trình bậc cao dùng cho automation, data analysis, ML và web APIs |
| 2 | How does chunking affect retrieval quality? | Chunk quá nhỏ mất context, chunk quá lớn pha loãng semantic relevance — cần cân bằng |
| 3 | What is the role of metadata in vector search? | Metadata giúp lọc và thu hẹp search space, cải thiện precision, tránh lấy sai loại tài liệu |
| 4 | How does cosine similarity work for text embeddings? | Đo góc giữa 2 vector, bất biến với độ dài vector, trả về 1.0 nếu giống hệt nhau |
| 5 | What are common failure cases in retrieval systems? | Tài liệu cũ xếp hạng cao, chunk quá nhỏ mất caveat, embedding model không xử lý tốt nội dung đa ngôn ngữ |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is Python and what is it used for? | python_intro.txt_0: "Python is a high-level programming language..." | 0.7985 | **Có** (chính xác) | Python dùng cho automation, data analysis, ML và web APIs |
| 2 | How does chunking affect retrieval quality? | vector_store_notes.md_4: "The quality of the retrieval system depends heavily on chunks..." | 0.6178 | **Có** (chính xác) | Chunk quá nhỏ mất context, quá lớn pha loãng semantic relevance |
| 3 | What is the role of metadata in vector search? | vector_store_notes.md_2: "A common vector search pipeline has four stages..." | 0.6089 | **Có** (liên quan) | Metadata giúp lọc search space, tăng precision |
| 4 | How does cosine similarity work for text embeddings? | vector_store_notes.md_3: "Embed the query and rank stored vectors by similarity..." | 0.4434 | **Có** (liên quan) | Đo góc giữa 2 vector, bất biến với độ dài |
| 5 | What are common failure cases in retrieval systems? | rag_system_design.md_11: "Example failure cases might include outdated documents..." | 0.5640 | **Có** (chính xác) | Tài liệu cũ xếp hạng cao, chunk nhỏ mất caveat, nội dung đa ngôn ngữ |

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| cmmi_btl_nhom2.txt | SentenceChunker (best baseline) | 316 | 339 | Tốt — giữ câu nguyên vẹn nhưng chunk ngắn |
| cmmi_btl_nhom2.txt | **RecursiveChunker (của tôi)** | 282 | 381 | Tốt hơn — giữ đoạn văn, chunk dài hơn, ngữ cảnh đầy đủ hơn |
| cmmi_v20_full_model.txt | SentenceChunker (best baseline) | 1715 | 485 | Tốt — nhưng số lượng chunk lớn |
| cmmi_v20_full_model.txt | **RecursiveChunker (của tôi)** | 1987 | 421 | Tốt — tôn trọng cấu trúc tài liệu gốc |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Lâm Văn Tài) | RecursiveChunker | 8 | Giữ ngữ cảnh đoạn văn, tôn trọng cấu trúc tài liệu | Chunk count cao hơn, tốn bộ nhớ hơn |
| [Tên thành viên 2] | FixedSizeChunker | — | Đơn giản, dễ kiểm soát | Cắt giữa câu, mất ngữ cảnh |
| [Tên thành viên 3] | SentenceChunker | — | Giữ nguyên câu hoàn chỉnh | Chunk ngắn, thiếu ngữ cảnh đoạn văn |


**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

> **Nhận xét:** Sử dụng **Local Embedder (all-MiniLM-L6-v2)** cho kết quả vượt trội so với Mock Embedder — 5/5 queries trả về đúng document nguồn, score cao nhất đạt 0.7985. Điều này cho thấy sức mạnh của semantic embedding thật sự so với hash-based mock.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> SentenceChunker bảo toàn câu hoàn chỉnh và tốt hơn cho các domain viết văn xuôi rõ ràng. Tuy nhiên, nó tạo ra chunk dài hơn nhiều, không phải lúc nào cũng tốt hơn cho retrieval khi query ngắn gọn. Điều này dạy tôi rằng không có một strategy "tốt nhất" tuyệt đối — phụ thuộc vào cả query pattern và document type.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Metadata filtering có thể cải thiện đáng kể precision khi bộ tài liệu lớn và đa dạng — đặc biệt khi người dùng có thể chỉ định ngôn ngữ hoặc loại tài liệu trong câu hỏi. Không nên bỏ qua metadata design khi thiết kế hệ thống RAG.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tăng cường thêm metadata (ví dụ: thêm trường `topic` và `difficulty_level`) để hỗ trợ pre-filtering hiệu quả hơn. Ngoài ra sẽ thử nghiệm thêm overlap trong RecursiveChunker để tránh mất context tại ranh giới chunk — đặc biệt quan trọng cho tài liệu kỹ thuật có nhiều thuật ngữ liên kết nhau.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **86 / 100** |
