from utils.database import get_db
from sqlalchemy import text
db = next(get_db())

result = db.execute(text("""
SELECT COUNT(*)
FROM media_files
""")).scalar()

print("MEDIA FILE COUNT =", result)