from fastapi import FastAPI
from src.backend.api import router

app = FastAPI()
app.include_router(router)

# def main():

# if __name__ == "__main__":
#     main()
