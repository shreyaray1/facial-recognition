import os
 
import psycopg2
from imgbeddings import imgbeddings
from pgvector.psycopg2 import register_vector
from PIL import Image
 
QUERY_DIR = "stored-faces"
 
# Cosine distance: 0 = identical direction, bigger = less similar.
# This is a STARTING GUESS. Print distances for known-correct and
# known-wrong matches and adjust it based on what you see.
THRESHOLD = 0.30
 
conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)
cur = conn.cursor()
 
ibed = imgbeddings()
 
for filename in sorted(os.listdir(QUERY_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
 
    path = os.path.join(QUERY_DIR, filename)
    img = Image.open(path).convert("RGB")
    embedding = ibed.to_embeddings(img)[0]
 
    cur.execute(
        """
        SELECT actor_name, embedding <=> %s AS distance
        FROM actor_faces
        ORDER BY distance
        LIMIT 1;
        """,
        (embedding,),
    )
    row = cur.fetchone()
 
    if row is None:
        print(f"{filename}: no reference faces in database")
        continue
 
    actor_name, distance = row
    if distance <= THRESHOLD:
        print(f"{filename}: {actor_name} (distance {distance:.3f})")
    else:
        print(f"{filename}: unknown (closest was {actor_name}, distance {distance:.3f})")
 
cur.close()
conn.close()