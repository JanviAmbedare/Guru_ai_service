import os


class FileUtils:

    @staticmethod
    def delete_file(file_path):

        try:

            if os.path.exists(file_path):

                os.remove(file_path)

                return True

            return False

        except Exception:

            return False