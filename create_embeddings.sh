#PICKLE_FILE=all_datasets_v3_mpnet-base.pkl EMBEDDINGS_MODEL_NAME=flax-sentence-embeddings/all_datasets_v3_mpnet-base python create_embeddings.py > datasets_v3_mpnet-base.log
#PICKLE_FILE=all-MiniLM-L6-v2.pkl EMBEDDINGS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 python create_embeddings.py > MiniLM-L6-v2.log
#PICKLE_FILE=all-mpnet-base-v2.pkl EMBEDDINGS_MODEL_NAME=sentence-transformers/all-mpnet-base-v2 python create_embeddings.py > mpnet-base-v2.log
#PICKLE_FILE=e5-large-v2.pkl EMBEDDINGS_MODEL_NAME=intfloat/e5-large-v2 python create_embeddings.py > e5-large-v2.log
PICKLE_FILE=e5-large-v2-ct2 EMBEDDINGS_MODEL_NAME=intfloat/e5-large-v2 python create_embeddings.py > e5-large-v2-ct2.log
