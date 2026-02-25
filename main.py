from fastapi import FastAPI
from src.backend.api import router

#def main():
app = FastAPI()
app.include_router(router)


# if __name__ == "__main__":
#     main()
