#!/usr/bin/env python3

# Libraries ---
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from global_utilities.keys import get_env_key
from global_utilities.paths import CANNONDALE_BIKES_ASSISTANT_DIR

def main():
    print("🚀 Starting vectorstore creation...")
    
    # Constants ----
    OPENAI_API_KEY = get_env_key("openai")
    EMBEDDING_MODEL = "text-embedding-ada-002"
    VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"

    # Load Scraped Data ----
    csv_path = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "csv" / "bikes_with_beautifulsoup_final.csv"
    print(f"📊 Loading data from: {csv_path}")

    bikes_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(bikes_df)} bikes from CSV")

    bikes_df = bikes_df.rename(columns={"platform": "bike_name", "model_name": "bike_model"})

    bikes_dict = bikes_df.to_dict(orient='records')

    documents = []

    for i, item in enumerate(bikes_dict):
        content = f"""
        bike_name: {item.get("bike_name")}
        bike_model: {item.get("bike_model")}
        price: {item.get("price")}
        sale_price: {item.get("sale_price")}
        color: {item.get("color")}
        description_1: {item.get("description_1")}
        description_2: {item.get("description_2")}
        description_3: {item.get("description_3")}
        highlights: {item.get("highlights")}
        model_code: {item.get("model_code")}
        frame: {item.get("frame")}
        fork: {item.get("fork")}
        headset: {item.get("headset")}
        rear_derailleur: {item.get("rear_derailleur")}
        front_derailleur: {item.get("front_derailleur")}
        shifters: {item.get("shifters")}
        chain: {item.get("chain")}
        crank: {item.get("crank")}
        rear_cogs: {item.get("rear_cogs")}
        bottom_bracket: {item.get("bottom_bracket")}
        brakes: {item.get("brakes")}
        brake_levers: {item.get("brake_levers")}
        front_hub: {item.get("front_hub")}
        rear_hub: {item.get("rear_hub")}
        rims: {item.get("rims")}
        spokes: {item.get("spokes")}
        tire_size: {item.get("tire_size")}
        wheel_size: {item.get("wheel_size")}
        tires: {item.get("tires")}
        front_tire: {item.get("front_tire")}
        rear_tire: {item.get("rear_tire")}
        handlebar: {item.get("handlebar")}
        stem: {item.get("stem")}
        grips: {item.get("grips")}
        saddle: {item.get("saddle")}
        seatpost: {item.get("seatpost")}
        wheel_sensor: {item.get("wheel_sensor")}
        extra_1: {item.get("extra_1")}
        bike_image_url: {item.get("bike_image_url")}
        hubs: {item.get("hubs")}
        ingestion_hazard: {item.get("ingestion_hazard")}
        rear_shock: {item.get("rear_shock")}
        drive_unit: {item.get("drive_unit")}
        battery: {item.get("battery")}
        charger: {item.get("charger")}
        display: {item.get("display")}
        certifications: {item.get("certifications")}
        brake_type: {item.get("brake_type")}
        """

        doc = Document(page_content=content, metadata=item)
        documents.append(doc)
        
        if (i + 1) % 50 == 0:
            print(f"📄 Processed {i + 1} documents...")

    print(f"✅ Created {len(documents)} documents")

    # Sample check
    if documents:
        print(f"🔍 Sample metadata keys: {list(documents[0].metadata.keys())}")
        sample_url = documents[0].metadata.get('bike_image_url', 'No URL')
        print(f"🖼️  Sample image URL: {sample_url[:100]}..." if len(str(sample_url)) > 100 else f"🖼️  Sample image URL: {sample_url}")

    # ------------------------------------------------------------------------------
    # VECTOR DATABASE ---
    # ------------------------------------------------------------------------------

    print("🔧 Creating embedding function...")
    # Embedding Function ----
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=OPENAI_API_KEY
    )

    print("🗄️  Creating vectorstore...")
    # Vector Database ----
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(VECTORSTORE_PATH),
        collection_name="bikes"
    )

    print("💾 Persisting vectorstore...")
    vectorstore.persist()

    print("✅ Vectorstore created successfully!")

    # ------------------------------------------------------------------------------
    # TEST RETRIEVER ----
    # ------------------------------------------------------------------------------

    print("🔍 Testing retriever...")
    # Retriever ----
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    test_queries = ["SuperSix EVO", "mountain bike", "road bike"]
    for query in test_queries:
        results = retriever.invoke(query)
        print(f"📋 Query '{query}': {len(results)} results")
        if results:
            metadata = results[0].metadata
            bike_name = metadata.get('bike_name', 'Unknown')
            bike_model = metadata.get('bike_model', 'Unknown')
            image_url = metadata.get('bike_image_url', 'No image')
            has_image = 'Yes' if image_url and str(image_url).startswith('http') else 'No'
            print(f"   - First result: {bike_name} {bike_model} (Image: {has_image})")

    print("🎉 All done!")

if __name__ == "__main__":
    main()
