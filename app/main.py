from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import app.settings

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline

app = FastAPI(title="Refiner and Reranker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    """
    Refinement agent loop:
    - Iteratively refines user query
    - Calls search pipeline to evaluate results
    - Terminates based on result quality or max iterations
    """
    agent = QueryRefinerAgent()
    return await agent.refine(request)


@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    """
    Vector retrieval + filtering + re-ranking pipeline:
    - Recalls top_k_raw from Qdrant
    - Post-filters by metadata (geography, etc.)
    - Re-ranks and returns top_k_final
    - Provides diagnostics
    """
    pipeline = SearchPipeline()
    return await pipeline.run(request)

