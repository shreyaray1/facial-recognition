# Facial Recognition

A small Python project that detects faces in a photo and identifies them by comparing face embeddings against a labeled reference set stored in PostgreSQL with pgvector. Built as a learning project using the Brooklyn Nine-Nine cast.

Based on Aiven's tutorial, [Image recognition with Python, OpenCV, OpenAI CLIP and pgvector](https://aiven.io/developer/find-faces-with-pgvector).

## Result

On a test photo of the cast, the system correctly identified **5 of 6 faces**.

## How it works

1. **Detect:** OpenCV's Haar cascade finds faces in a photo and crops each one.
2. **Embed:** `imgbeddings` (OpenAI's CLIP model via Hugging Face) turns each cropped face into a 768-dimensional vector.
3. **Store:** reference embeddings for each known person are stored in PostgreSQL (hosted on Aiven) using the pgvector extension.
4. **Match:** for each face in the target photo, a nearest-neighbor query using cosine distance finds the closest reference face. If the distance is above a threshold, the face is labeled unknown.

## Files

| File | Purpose |
| --- | --- |
| `crop_reference.py` | Crops faces from downloaded reference photos into `reference-faces/<name>/` |
| `store_embeddings.py` | Embeds the reference faces and stores them in the database (run once) |
| `extract-faces.py` | Detects and crops faces from the target photo into `stored-faces/` |
| `match-faces.py` | Embeds each cropped face and finds its closest match |

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Create a PostgreSQL service on [Aiven](https://aiven.io) and copy its Service URI.
3. Set it as an environment variable:
   ```
   export DATABASE_URL="postgres://USER:PASSWORD@HOST:PORT/defaultdb?sslmode=require"
   ```
4. Put reference photos in `reference-photos/<name>/` (one person per folder, one face per photo), then run:
   ```
   python3 crop_reference.py
   python3 store_embeddings.py
   ```
5. Run face detection on the target photo, then match:
   ```
   python3 extract-faces.py
   python3 match-faces.py
   ```

The `.jpg` files and image folders are not included in this repository.

## Limitations

- This is a learning project, not a production face recognition system.
- Haar cascades can miss faces or detect false ones, and the reference set is small, so accuracy depends heavily on photo quality, angle, and lighting.
- The distance threshold was tuned by hand on a single test photo.
- CLIP is a general-purpose image model rather than one trained specifically for face recognition, so dedicated face models would likely perform better.
