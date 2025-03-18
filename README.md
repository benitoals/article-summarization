# Abstractive summarization of scientific articles
Large volumes of domain-specific text can hinder efficient information extraction in scientific and cyber intelligence contexts. This study explores an abstractive summarization approach that employs a relatively small, adapter-based model (mT5-base with LoRA fine-tuning) to handle literature related to specialized industrial control systems. We evaluate a baseline, pre-trained model on a locally compiled dataset of 73 scientific articles. Our findings suggest that initially incorporating broad scientific knowledge—even if imperfectly aligned—followed by targeted local domain adaptation can yield better summaries compared to relying solely on a small domain corpus. Despite these improvements, human evaluation indicates incomplete summaries that overlook critical elements from reference abstracts, revealing opportunities for further enhancement. We conclude that LoRA-based fine-tuning is an effective and resource-efficient strategy for abstractive summarization in specialized domains. Future work may focus on larger models, expanded domain corpora, and improved preprocessing techniques to enhance summary coherence and completeness. Ultimately, this approach proves especially valuable where rapid and accurate text summarization is essential, such as in time-sensitive or resource-constrained cyber intelligence operations.