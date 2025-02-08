from mangum import Mangum
from app import app

# Create the handler
handler = Mangum(app, lifespan="off") 