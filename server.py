import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 4000)), # You can always change the port number as you want. This is for local server.
        access_log=True,
        reload=True,
    )