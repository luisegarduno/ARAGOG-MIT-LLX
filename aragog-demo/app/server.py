from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import sys
import os

# Add package to path
currentFilePath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentFilePath)

# Import chain
from rag_chroma_private_llsc_chain import chain

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
