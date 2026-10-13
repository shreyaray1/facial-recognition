import os
 
import psycopg2
from imgbeddings import imgbeddings
from pgvector.psycopg2 import register_vector
from PIL import Image
 
REFERENCE_DIR = "reference-faces"
 
# Aiven gives you a "Service URI" in the console; export it as an env variable:
#   export DATABASE_URL="postgres://avnadmin:PASSWORD@HOST:PORT/defaultdb?sslmode=require"
conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
 
# the vector extension must exist BEFORE register_vector() is called
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
register_vector(conn)
 
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS actor_faces (
        id SERIAL PRIMARY KEY,
        actor_name TEXT NOT NULL,
        filename TEXT NOT NULL,
        embedding vector(768)
    );
    """
)
conn.commit()
 
ibed = imgbeddings()
 
count = 0
for actor_name in sorted(os.listdir(REFERENCE_DIR)):
    actor_dir = os.path.join(REFERENCE_DIR, actor_name)
    if not os.path.isdir(actor_dir):
        continue
 
    for filename in sorted(os.listdir(actor_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
 
        path = os.path.join(actor_dir, filename)
        # CLIP expects RGB, even if the crop was saved as grayscale
        img = Image.open(path).convert("RGB")
        embedding = ibed.to_embeddings(img)[0]  # shape (768,)
 
        cur.execute(
            "INSERT INTO actor_faces (actor_name, filename, embedding) VALUES (%s, %s, %s);",
            (actor_name, filename, embedding),
        )
        count += 1
        print(f"Stored {actor_name}/{filename}")
 
conn.commit()
cur.close()
conn.close()
print(f"Done. Inserted {count} reference faces.")