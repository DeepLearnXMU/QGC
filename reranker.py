def get_distance_bm25(corpus, query):
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    idx = [(ii, for ii in -doc_scores.argsort()]
    return idx


def get_rank_results(
    self,
    context: list,
    question: str,
    rank_method: str,
    condition_in_question: str,
    context_tokens_length: list,
):
    def get_distance_bm25(corpus, query):
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        idx = [(ii, 0) for ii in (-doc_scores).argsort()]
        return idx

    def get_distance_gzip(corpus, query):
        def get_score(x, y):
            cx, cy = len(gzip.compress(x.encode())), len(gzip.compress(y.encode()))
            cxy = len(gzip.compress(f"{x} {y}".encode()))
            return (cxy - min(cx, cy)) / max(cx, cy)

        import gzip

        doc_scores = [get_score(doc, query) for doc in corpus]
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_sentbert(corpus, query):
        from sentence_transformers import SentenceTransformer, util

        if self.retrieval_model is None or self.retrieval_model_name != rank_method:
            self.retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            self.retrieval_model_name = rank_method
        doc_embeds = self.retrieval_model.encode(corpus)
        query = self.retrieval_model.encode(query)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_openai(corpus, query):
        import openai
        from sentence_transformers import util

        openai.api_key = self.open_api_config.get("api_key", "")
        openai.api_base = self.open_api_config.get(
            "api_base", "https://api.openai.com/v1"
        )
        openai.api_type = self.open_api_config.get("api_type", "open_ai")
        openai.api_version = self.open_api_config.get("api_version", "2023-05-15")
        engine = self.open_api_config.get("engine", "text-embedding-ada-002")

        def get_embed(text):
            return openai.Embedding.create(
                input=[text.replace("\n", " ")], engine=engine
            )["data"][0]["embedding"]

        doc_embeds = [get_embed(i) for i in corpus]
        query = get_embed(query)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_sentbert_bge(corpus, query):
        from sentence_transformers import SentenceTransformer, util

        if self.retrieval_model is None or self.retrieval_model_name != rank_method:
            self.retrieval_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            self.retrieval_model_name = rank_method
        doc_embeds = self.retrieval_model.encode(
            [i for i in corpus], normalize_embeddings=True
        )
        query = self.retrieval_model.encode(query, normalize_embeddings=True)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_bge_ranker(corpus, query):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pairs = [[i, query] for i in corpus]
        if self.retrieval_model is None or self.retrieval_model_name != rank_method:
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    "BAAI/bge-reranker-large"
                )
                .eval()
                .to(self.device)
            )
            self.retrieval_model = [tokenizer, model]
            self.retrieval_model_name = rank_method
        with torch.no_grad():
            inputs = self.retrieval_model[0](
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            scores = (
                self.retrieval_model[1](**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
        idx = [(ii, 0) for ii in np.argsort(-scores.cpu())]
        return idx

    def get_distance_bge_llmembedder(corpus, query):
        from transformers import AutoModel, AutoTokenizer

        if self.retrieval_model is None or self.retrieval_model_name != rank_method:
            tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
            model = (
                AutoModel.from_pretrained("BAAI/llm-embedder")
                .eval()
                .to(self.device)
            )
            self.retrieval_model = [tokenizer, model]
            self.retrieval_model_name = rank_method

        instruction_qa_query = (
            "Represent this query for retrieving relevant documents: "
        )
        instruction_qa_key = "Represent this document for retrieval: "
        queries = [instruction_qa_query + query for _ in corpus]
        keys = [instruction_qa_key + key for key in corpus]
        with torch.no_grad():
            query_inputs = self.retrieval_model[0](
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            key_inputs = self.retrieval_model[0](
                keys,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            query_outputs = self.retrieval_model[1](**query_inputs)
            key_outputs = self.retrieval_model[1](**key_inputs)
            # CLS pooling
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            key_embeddings = key_outputs.last_hidden_state[:, 0]
            # Normalize
            query_embeddings = torch.nn.functional.normalize(
                query_embeddings, p=2, dim=1
            )
            key_embeddings = torch.nn.functional.normalize(
                key_embeddings, p=2, dim=1
            )
            similarity = query_embeddings @ key_embeddings.T
        idx = [(ii, 0) for ii in np.argsort(-similarity[0].cpu())]
        return idx

    def get_distance_jinza(corpus, query):
        from numpy.linalg import norm

        from transformers import AutoModel

        def cos_sim(a, b):
            return (a @ b.T) / (norm(a) * norm(b))

        if self.retrieval_model is None or self.retrieval_model_name != rank_method:
            model = (
                AutoModel.from_pretrained(
                    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
                )
                .eval()
                .to(self.device)
            )
            self.retrieval_model = model
            self.retrieval_model_name = rank_method

        doc_embeds = self.retrieval_model.encode(corpus)
        query = self.retrieval_model.encode(query)
        doc_scores = cos_sim(doc_embeds, query)
        idx = [(ii, 0) for ii in np.argsort(-doc_scores)]
        return idx

    def get_distance_voyageai(corpus, query):
        import voyageai
        from sentence_transformers import util

        voyageai.api_key = self.open_api_config.get("voyageai_api_key", "")

        def get_embed(text):
            return voyageai.get_embedding(text, model="voyage-01")

        doc_embeds = [get_embed(i) for i in corpus]
        query = get_embed(query)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_cohere(corpus, query):
        import cohere

        api_key = self.open_api_config.get("cohere_api_key", "")
        co = cohere.Client(api_key)
        results = co.rerank(
            model="rerank-english-v2.0", query=query, documents=corpus, top_n=20
        )
        c_map = {jj: ii for ii, jj in enumerate(corpus)}
        doc_rank = [c_map[ii.document["text"]] for ii in results]
        idx = [(ii, 0) for ii in doc_rank]
        return idx

    def get_distance_longllmlingua(corpus, query):
        context_ppl = [
            self.get_condition_ppl(
                d,
                query
                + " We can get the answer to this question in the given documents.",
                condition_in_question,
            )
            - dl * 2 / 250 * 0
            for d, dl in zip(corpus, context_tokens_length)
        ]
        sort_direct = -1 if condition_in_question == "none" else 1
        ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
        return ys

    method = None
    if rank_method == "bm25":
        method = get_distance_bm25
    elif rank_method == "gzip":
        method = get_distance_gzip
    elif rank_method == "sentbert":
        method = get_distance_sentbert
    elif rank_method == "openai":
        method = get_distance_openai
    elif rank_method in ["longllmlingua", "llmlingua"]:
        method = get_distance_longllmlingua
    elif rank_method == "bge":
        method = get_distance_sentbert_bge
    elif rank_method == "bge_reranker":
        method = get_distance_bge_ranker
    elif rank_method == "bge_llmembedder":
        method = get_distance_bge_llmembedder
    elif rank_method == "jinza":
        method = get_distance_jinza
    elif rank_method == "voyageai":
        method = get_distance_voyageai
    elif rank_method == "cohere":
        method = get_distance_cohere
    return method(context, question)