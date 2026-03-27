from fastapi import FastAPI

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline

app = FastAPI(title="Refiner and Reranker")
_refiner_agent = QueryRefinerAgent()
_search_pipeline = SearchPipeline()


@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    return await _refiner_agent.refine(request)


@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    return await _search_pipeline.run(request)
