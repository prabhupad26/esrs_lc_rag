import os


def initialize_env(env_configs):
    local_inference = env_configs.get("local_inference", {})
    env_variables = local_inference.get("env_variables", {})

    for key, value in env_variables.items():
        os.environ[key] = value


def format_docs(docs):
    return "\n\n".join(f"[{doc_idx}]: {doc.page_content}" for doc_idx, doc in enumerate(docs))
