
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from routers.ocr import router


app = FastAPI(
    title="Medical Reports OCR Backend",
    description=(
        "OCR API for printed lab reports and medical reports, "
        "with structured field extraction and chronological sorting."
    ),
    version="0.2.0",
)

app.include_router(router, prefix="/api/v1")


def custom_openapi():
    """Make Swagger display uploaded files as file pickers."""

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.3",
    )

    properties_by_schema = (
        schema.get("components", {})
        .get("schemas", {})
        .values()
    )

    for model_schema in properties_by_schema:
        properties = model_schema.get("properties", {})

        for name, field_schema in properties.items():

            # Fix multiple-file upload fields.
            if name == "files" and field_schema.get("type") == "array":
                field_schema["items"] = {
                    "type": "string",
                    "format": "binary",
                }

            # Fix single-file upload fields too.
            elif name == "file" and field_schema.get("type") == "string":
                field_schema.clear()
                field_schema.update({
                    "type": "string",
                    "format": "binary",
                })

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def root():
    return {
        "message": "VedAmrit Medical OCR Backend is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
