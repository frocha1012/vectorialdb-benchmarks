"""
Diagnostic script to check MongoDB Atlas vector search configuration.
Run this to diagnose why recall is 0.0.
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def main():
    connection_string = os.getenv("MONGODB_ATLAS_URI")
    if not connection_string:
        print("❌ Error: MONGODB_ATLAS_URI environment variable not set")
        print("   Set it in your .env file or environment")
        return
    
    print("MongoDB Atlas Vector Search Diagnostic")
    print("=" * 60)
    
    try:
        # Connect to MongoDB
        client = MongoClient(connection_string)
        db = client["vectordb"]
        collection = db["documents"]
        
        print(f"\n1. Connection Status: Connected")
        print(f"   Database: vectordb")
        print(f"   Collection: documents")
        
        # Check document count
        doc_count = collection.count_documents({})
        print(f"\n2. Document Count: {doc_count}")
        if doc_count == 0:
            print("   WARNING: Collection is empty!")
            print("   → Data needs to be inserted before search can work")
            return
        else:
            print(f"   Collection has {doc_count} documents")
        
        # Check a sample document
        sample = collection.find_one({})
        if sample:
            print(f"\n3. Sample Document Structure:")
            print(f"   _id: {sample.get('_id')}")
            print(f"   text: {sample.get('text', 'N/A')[:50]}...")
            if 'embedding' in sample:
                emb = sample['embedding']
                if isinstance(emb, list):
                    print(f"   embedding: List with {len(emb)} dimensions")
                    print(f"   Embedding field exists")
                else:
                    print(f"   embedding: {type(emb)}")
                    print(f"   WARNING: Embedding is not a list")
            else:
                print(f"   ERROR: No 'embedding' field found!")
        
        # Check for vector search index
        print(f"\n4. Vector Search Index:")
        print(f"   WARNING: Cannot programmatically check index existence")
        print(f"   Go to Atlas UI -> Atlas Search -> Search Indexes")
        print(f"   Look for index named 'vector_index'")
        print(f"   It should have:")
        print(f"      - path: 'embedding'")
        print(f"      - numDimensions: 384 (or your embedding dimension)")
        print(f"      - similarity: 'cosine'")
        
        # Try a test search
        print(f"\n5. Testing Vector Search:")
        if 'embedding' in sample and isinstance(sample['embedding'], list):
            test_vector = sample['embedding']  # Use existing vector as query
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": test_vector,
                        "numCandidates": 10,
                        "limit": 5,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            
            try:
                results = list(collection.aggregate(pipeline))
                if len(results) > 0:
                    print(f"   Vector search works! Returned {len(results)} results")
                    print(f"   Top result: id={results[0].get('_id')}, score={results[0].get('score', 'N/A'):.4f}")
                else:
                    print(f"   Vector search returned 0 results")
                    print(f"   This indicates the index 'vector_index' doesn't exist or is misconfigured")
                    print(f"   Action: Create the index in Atlas UI with the configuration shown above")
            except Exception as e:
                print(f"   Vector search failed with error:")
                print(f"      {str(e)}")
                print(f"   This confirms the index 'vector_index' is missing or misconfigured")
                print(f"   Action: Create the index in Atlas UI")
        else:
            print(f"   Skipped (no valid embedding field to test with)")
        
        print(f"\n" + "=" * 60)
        print("Summary:")
        if doc_count > 0:
            print("   Data exists in collection")
            print("   Check if 'vector_index' exists in Atlas Search")
            print("   If index doesn't exist, create it with:")
            print("      - Name: vector_index")
            print("      - Path: embedding")
            print("      - Dimensions: 384 (check your EmbeddingGenerator model)")
            print("      - Similarity: cosine")
        else:
            print("   No data in collection")
            print("   Run the benchmark to insert data first")
        
    except Exception as e:
        print(f"\nConnection Error: {e}")
        print("   Check your MONGODB_ATLAS_URI connection string")
        print("   Verify network access is enabled in Atlas")
        print("   Check username/password are correct")

if __name__ == "__main__":
    main()
