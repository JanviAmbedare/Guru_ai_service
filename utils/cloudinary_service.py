import os
import cloudinary
import cloudinary.uploader
import tempfile
import requests
import os
from dotenv import load_dotenv


load_dotenv()


cloudinary.config(
    cloud_name=os.getenv(
        "CLOUDINARY_CLOUD_NAME"
    ),

    api_key=os.getenv(
        "CLOUDINARY_API_KEY"
    ),

    api_secret=os.getenv(
        "CLOUDINARY_API_SECRET"
    )
)


class CloudinaryService:

    @staticmethod
    def download_file(url):

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        suffix = os.path.splitext(
            url.split("?")[0]
        )[1]

        if not suffix:
            suffix = ".tmp"

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_file.write(
            response.content
        )

        temp_file.close()

        return temp_file.name

    @staticmethod
    def upload_file(
        file_path,
        folder
    ):

        try:

            result = (
                cloudinary.uploader.upload(
                    file_path,
                    folder=folder,
                    resource_type="auto"
                )
            )

            return {
                "url": result["secure_url"],
                "public_id": (
                    result["public_id"]
                )
            }

        except Exception as e:

            raise Exception(
                f"Cloudinary upload failed: "
                f"{e}"
            )
    @staticmethod
    def upload_embedding(
        file_path,
        embedding_type,
        user_id
    ):

        result = cloudinary.uploader.upload(
            file_path,
            folder=(
                f"guru/embeddings/"
                f"{embedding_type}"
            ),
            resource_type="raw",
            public_id=str(user_id)
        )

        return {
            "url": result["secure_url"],
            "public_id": result["public_id"]
        }

    @staticmethod
    def delete_file(public_id):

        try:

            result = (
                cloudinary.uploader.destroy(
                    public_id,
                    resource_type="auto"
                )
            )

            return result

        except Exception as e:

            raise Exception(
                f"Cloudinary delete failed: "
                f"{e}"
            )