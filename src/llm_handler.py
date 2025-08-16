from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class QAPipeline:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.qa_pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # Force CPU
        )

    def build_prompt(self, chunks, question, context_max_length=1500):
        """
        Build a prompt from top-k retrieved chunks and user question.
        Truncate context if too long for model.
        """
        context = "\n\n".join(chunks)
        # Limit context to a reasonable size to fit model input (in tokens)
        if len(context) > context_max_length:
            context = context[:context_max_length]
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def get_answer(self, chunks, question):
        """
        Generate answer using top-k text chunks as context and user question.
        """
        prompt = self.build_prompt(chunks, question)
        answer = self.qa_pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        return answer.strip()
