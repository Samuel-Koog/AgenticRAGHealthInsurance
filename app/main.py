from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .service import (
    CitySearchRequest,
    ProximityCityRequest,
    SearchRequest,
    clean_results_compact,
    clean_results_compact_with_distance,
    query_by_city,
    query_by_zip,
    save_clean_compact,
)

app = FastAPI(title="NPI Search API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search_zip(req: SearchRequest):
    try:
        return await query_by_zip(req.zip, req.state, req.specialty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/city")
async def search_city(req: CitySearchRequest):
    try:
        return await query_by_city(req.city, req.state, req.specialty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/clean/compact")
async def search_clean_compact_zip(req: SearchRequest):
    try:
        raw = await query_by_zip(req.zip, req.state, req.specialty)
        cleaned = clean_results_compact(
            raw,
            req.specialty,
            scope_filter={"zip": req.zip, "state": (req.state or "")},
        )
        save_clean_compact(cleaned, f"{req.zip}_{req.specialty}")
        return cleaned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/city/clean/compact")
async def search_clean_compact_city(req: CitySearchRequest):
    try:
        raw = await query_by_city(req.city, req.state, req.specialty)
        cleaned = clean_results_compact(
            raw,
            req.specialty,
            scope_filter={"city": req.city, "state": req.state},
        )
        save_clean_compact(cleaned, f"{req.city.lower().replace(' ', '_')}_{req.state}_{req.specialty}")
        return cleaned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/city/near")
async def search_city_near(req: ProximityCityRequest):
    """
    City-wide search, but results are sorted by distance from origin_zip.
    """
    try:
        raw = await query_by_city(req.city, req.state, req.specialty)
        cleaned = clean_results_compact_with_distance(
            raw,
            req.specialty,
            origin_zip=req.origin_zip,
            scope_filter={"city": req.city, "state": req.state},
        )
        # Save for debugging
        save_clean_compact(cleaned, f"{req.city.lower().replace(' ', '_')}_{req.state}_{req.specialty}_near_{req.origin_zip}")
        return cleaned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))