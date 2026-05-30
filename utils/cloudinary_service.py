import os
import cloudinary
import cloudinary.uploader

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