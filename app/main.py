from fastapi import FastAPI

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline

app = FastAPI(title="Refiner and Reranker")

@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    agent = QueryRefinerAgent()
    return await agent.refine(request)

@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    pipeline = SearchPipeline()
    result = await pipeline.run(request)
    return result

