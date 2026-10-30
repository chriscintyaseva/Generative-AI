
import os
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from langchain_ollama import OllamaEmbeddings

# ==========================================
# 1. KONFIGURASI
# ==========================================

WEAVIATE_URL = "https://co4b4zicucbehzkldusr4vakqqq.c0.eu-central-1.aws.weaviate.cloud"

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

COLLECTION_NAME = "AIKnowledgeBase"

# Model embedding dari Ollama
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ==========================================
# 2. KONEKSI KE WEAVIATE CLOUD
# ==========================================

if not WEAVIATE_API_KEY:
    raise ValueError(
        "WEAVIATE_API_KEY belum diatur di environment variable."
    )

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
)

try:
    print("=" * 50)
    print("KONEKSI WEAVIATE BERHASIL")
    print("=" * 50)

    # ==========================================
    # 3. MEMBUAT COLLECTION
    # ==========================================

    if client.collections.exists(COLLECTION_NAME):
        collection = client.collections.get(COLLECTION_NAME)
        print("Collection sudah tersedia.")
    else:
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT
                )
            ]
        )
        print("Collection berhasil dibuat.")

    # ==========================================
    # 4. MENAMBAHKAN DATA
    # ==========================================

    documents = [
        {
            "title": "Artificial Intelligence",
            "content": "Artificial Intelligence adalah teknologi yang memungkinkan komputer meniru kecerdasan manusia."
        },
        {
            "title": "Machine Learning",
            "content": "Machine Learning merupakan cabang AI yang memungkinkan komputer belajar dari data."
        },
        {
            "title": "Deep Learning",
            "content": "Deep Learning menggunakan jaringan saraf tiruan dengan banyak lapisan untuk mempelajari pola data."
        }
    ]

    for index, document in enumerate(documents):
        vector = embedding_model.embed_query(document["content"])

        collection.data.insert(
            properties=document,
            vector=vector,
            uuid=f"00000000-0000-0000-0000-{index + 1:012d}"
        )

    print("Data berhasil dimasukkan.")

    # ==========================================
    # 5. MELAKUKAN VECTOR SEARCH
    # ==========================================

    query = "Apa itu kecerdasan buatan?"

    query_vector = embedding_model.embed_query(query)

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=3,
        return_metadata=MetadataQuery(distance=True)
    )

    print("\nHASIL VECTOR SEARCH")
    print("-" * 50)

    for item in results.objects:
        print("Judul:", item.properties["title"])
        print("Konten:", item.properties["content"])
        print("Distance:", item.metadata.distance)
        print("-" * 50)

    # ==========================================
    # 6. MENUTUP KONEKSI
    # ==========================================

finally:
    client.close()
    print("\nKoneksi Weaviate ditutup.")
