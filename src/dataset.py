import torch
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

instructions_map = {
    'base': 'Write a high-quality answer for the given question using only the provided search results(some of which might be irrelevant).\n\n',
    'short': 'Using only the provided search results (some of which might be irrelevant), answer the following question with one or few words.\n\n',
}

def format_document(document, tokenizer, max_tokens):
    return tokenizer.decode(
            tokenizer(
            document['title'] + ' ' + document['text'] if 'title' in document else document['text'],
            add_special_tokens=False,
        )['input_ids'][:max_tokens]
    )

class TrainDataset(Dataset):
    def __init__(
        self,
        filepath,
        enc_tokenizer,
        llm_tokenizer,
        max_doc_tokens,
        instruction_name,
        que_mask_ratio=None,
        max_num_documents=None,
        min_num_documents=None,
        random_num_documents=False,
        num_gold_documents=1,
        use_answer_as_target=False,
        gold_first_for_kd=False,
        **kwargs,
    ):
        self.dataset = load_dataset('json', data_files=filepath, split='train')
        self.max_doc_tokens = max_doc_tokens
        self.enc_tokenizer = enc_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.que_mask_ratio = que_mask_ratio
        self.max_num_documents = max_num_documents
        self.min_num_documents = min_num_documents
        self.random_num_documents = random_num_documents
        self.num_gold_documents = num_gold_documents
        self.use_answer_as_target = use_answer_as_target
        self.gold_first_for_kd = gold_first_for_kd

        self.llm_tokenizer.padding_side = 'left'
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = llm_tokenizer.unk_token
            self.llm_tokenizer.pad_token_id = llm_tokenizer.unk_token_id

        self.instruction_text = instructions_map[instruction_name]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        question = example['question']
        
        neg_documents = [
            format_document(document, self.enc_tokenizer, self.max_doc_tokens)
            for document in example['ctxs'] if document['isgold'] is False
        ]

        if len(neg_documents) > self.max_num_documents:
            neg_documents = random.sample(neg_documents, k = self.max_num_documents)
        else:
            random.shuffle(neg_documents)
            
        pos_documents = [
            format_document(document, self.enc_tokenizer, self.max_doc_tokens)
            for document in example['ctxs'] if document['isgold'] is True
        ]

        if len(pos_documents) > self.num_gold_documents:
            num_gold_documents = self.num_gold_documents
            if len(neg_documents) < self.max_num_documents:
                num_gold_documents = self.max_num_documents - len(neg_documents)
            pos_documents = random.sample(pos_documents, k = num_gold_documents)
        
        else:
            random.shuffle(pos_documents)

        if self.use_answer_as_target:
            appeared_answer_list = []
            for answer in example['answers']:
                if answer in '\n\n'.join(pos_documents):
                    appeared_answer_list.append(answer)

            target = random.choice(
                appeared_answer_list if appeared_answer_list != [] else example['answers']
            )
        else:
            target = example['target']

        answers = example['answers']

        return {
            'question': question,
            'neg_documents': neg_documents,
            'pos_documents': pos_documents,
            'target': target,
            'answers': answers,
        }

    def collate_fn(self, batch):
        if len(batch) == 0:
            return {}

        enc_documents = []
        llm_prefix_tokens = []
        llm_documents = []
        
        num_documents = (
            random.randint(self.min_num_documents, self.max_num_documents)
            if self.random_num_documents else self.max_num_documents
        )
        for instance in batch:
            instance_enc_documents = ['<DOC>' + document for document in instance['pos_documents'] + instance['neg_documents']][:num_documents]
            random.shuffle(instance_enc_documents)
            enc_documents += instance_enc_documents

            llm_candiate_documents = instance['pos_documents'] + instance['neg_documents']
            llm_candiate_documents = llm_candiate_documents[:num_documents]
            if not self.gold_first_for_kd:
                random.shuffle(llm_candiate_documents)

            llm_documents += [''.join(['\nDocument:' + document for document in llm_candiate_documents])]

        enc_questions = ['<QUE>' + instance['question'] for instance in batch]
        llm_questions = ['\nQuestion:' + instance['question'] + '\nAnswer:' for instance in batch]
        llm_targets = [instance['target'] for instance in batch]
        llm_instructions = [self.instruction_text for _ in batch]
        answers = [instance['answers'] for instance in batch]

        llm_prefix_tokens = ['\nDocument:' for _ in enc_documents]
        enc_que_outputs = self.enc_tokenizer(enc_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_doc_outputs = self.enc_tokenizer(enc_documents, return_tensors='pt', padding=True, add_special_tokens=False)

        llm_ins_outputs = self.llm_tokenizer(llm_instructions, return_tensors='pt', padding=True)
        llm_doc_outputs = self.llm_tokenizer(llm_documents, return_tensors='pt', padding=True, add_special_tokens=False)
        llm_que_outputs = self.llm_tokenizer(llm_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        llm_pfx_outputs = self.llm_tokenizer(llm_prefix_tokens, return_tensors='pt', padding=True, add_special_tokens=False)

        def right_padding(value, padding_value):
            padded_value = pad_sequence(
                [torch.tensor(v) for v in value],
                batch_first=True,
                padding_value=padding_value,
            )
            return padded_value

        llm_tgt_outputs = [self.llm_tokenizer(ans, add_special_tokens=False).input_ids for ans in llm_targets]
        llm_tgt_tokens = right_padding(llm_tgt_outputs, self.llm_tokenizer.pad_token_id)
        llm_tgt_mask = right_padding([[1] * len(elem) for elem in llm_tgt_outputs], 0)

        if self.que_mask_ratio is not None and self.que_mask_ratio > 0:
            llm_que_tokens = llm_que_outputs.input_ids
            random_indices = torch.rand_like(llm_que_outputs.input_ids[:, :-2].float()).sort().indices
            mask_indices = random_indices[:, :int(self.que_mask_ratio * llm_que_tokens.size(1))]
            llm_que_outputs.input_ids = llm_que_tokens.scatter(1, mask_indices, self.llm_tokenizer.unk_token_id)
        
        return {
            'enc_doc_tokens': enc_doc_outputs.input_ids,
            'enc_que_tokens': enc_que_outputs.input_ids,
            'enc_doc_mask': enc_doc_outputs.attention_mask,
            'enc_que_mask': enc_que_outputs.attention_mask,
            'llm_ins_tokens': llm_ins_outputs.input_ids,
            'llm_doc_tokens': llm_doc_outputs.input_ids,
            'llm_que_tokens': llm_que_outputs.input_ids,
            'llm_ins_mask': llm_ins_outputs.attention_mask,
            'llm_doc_mask': llm_doc_outputs.attention_mask,
            'llm_que_mask': llm_que_outputs.attention_mask,
            'llm_tgt_tokens': llm_tgt_tokens,
            'llm_tgt_mask': llm_tgt_mask,
            'llm_pfx_tokens': llm_pfx_outputs.input_ids,
            'llm_pfx_mask': llm_pfx_outputs.attention_mask,
            'answers': answers,
        }
    

class InferDataset(Dataset):
    def __init__(
        self,
        filepath,
        enc_tokenizer,
        llm_tokenizer,
        max_doc_tokens,
        max_num_documents=None,
        instruction_text='base',
        **kwargs,
    ):
        self.dataset = load_dataset('json', data_files=filepath, split='train')
        self.max_doc_tokens = max_doc_tokens
        self.enc_tokenizer = enc_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_num_documents = max_num_documents

        self.llm_tokenizer.padding_side = 'left'
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = llm_tokenizer.unk_token
            self.llm_tokenizer.pad_token_id = llm_tokenizer.unk_token_id

        self.instruction_text = instruction_text


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        example = self.dataset[index]
        question = example['question']
        
        documents = [
            format_document(document, self.enc_tokenizer, self.max_doc_tokens)
            for document in example['ctxs'][:self.max_num_documents]
        ]

        if len(documents) < self.max_num_documents:
            documents += ['\n' for _ in range(self.max_num_documents - len(documents))]

        answers = example['answers']

        return {
            'question': question,
            'documents': documents,
            'answers': answers,
        }


    def collate_fn(self, batch):
        if len(batch) == 0:
            return {}

        enc_documents = []
        llm_prefix_tokens = []
        for instance in batch:
            instance_documents = instance['documents']
            instance_enc_docuemnts = ['<DOC>' + document for document in instance_documents]
            enc_documents += instance_enc_docuemnts

        enc_questions = ['<QUE>' + instance['question'] for instance in batch]
        llm_prefix_tokens = [f'\nDocument:' for instance in batch for _ in instance['documents']]
        llm_questions = ['\nQuestion:' + instance['question'] + '\nAnswer:' for instance in batch]
        llm_instructions = [self.instruction_text for _ in batch]
        answers = [instance['answers'] for instance in batch]

        enc_que_outputs = self.enc_tokenizer(enc_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_doc_outputs = self.enc_tokenizer(enc_documents, return_tensors='pt', padding=True, add_special_tokens=False)

        llm_ins_outputs = self.llm_tokenizer(llm_instructions, return_tensors='pt', padding=True)
        llm_que_outputs = self.llm_tokenizer(llm_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        llm_pfx_outputs = self.llm_tokenizer(llm_prefix_tokens, return_tensors='pt', padding=True, add_special_tokens=False)

        return {
            'enc_doc_tokens': enc_doc_outputs.input_ids,
            'enc_que_tokens': enc_que_outputs.input_ids,
            'enc_doc_mask': enc_doc_outputs.attention_mask,
            'enc_que_mask': enc_que_outputs.attention_mask,
            'llm_ins_tokens': llm_ins_outputs.input_ids,
            'llm_que_tokens': llm_que_outputs.input_ids,
            'llm_ins_mask': llm_ins_outputs.attention_mask,
            'llm_que_mask': llm_que_outputs.attention_mask,
            'llm_pfx_tokens': llm_pfx_outputs.input_ids,
            'llm_pfx_mask': llm_pfx_outputs.attention_mask,
            'answers': answers
        }