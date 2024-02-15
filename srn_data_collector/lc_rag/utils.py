import os


def initialize_env(env_configs):
    local_inference = env_configs.get("local_inference", {})
    env_variables = local_inference.get("env_variables", {})

    for key, value in env_variables.items():
        os.environ[key] = value


def format_docs(docs):
    return "\n\n".join(f"[{doc_idx}]: {doc.page_content}" for doc_idx, doc in enumerate(docs))


def convert_file_name(original_filename):
    directory, file_with_extension = os.path.split(original_filename)
    filename, extension = os.path.splitext(file_with_extension)
    new_filename = filename + "_processed" + extension
    new_path = os.path.join(directory, new_filename)
    return new_path


def get_doc_name(original_filename):
    directory, file_with_extension = os.path.split(original_filename)
    filename, extension = os.path.splitext(file_with_extension)
    return filename


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["blob_id"] = record.get("blob_id")
    metadata["doc_ref"] = record.get("doc_ref")
    return metadata
