from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

async def pna_middleware_logic(request, call_next):
    # Intercept OPTIONS requests to inject PNA headers immediately
    # This must run BEFORE CORSMiddleware to prevent it from swallowing the request
    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        response = await call_next(request)
    
    # Inject PNA + CORP headers into ALL responses
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    
    # Ensure CORS headers are present even if CORSMiddleware missed them (double safety)
    if "Access-Control-Allow-Origin" not in response.headers:
        origin = request.headers.get("Origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE, PUT"
            response.headers["Access-Control-Allow-Headers"] = "*"
            
    return response
