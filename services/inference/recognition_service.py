from sklearn.metrics.pairwise import (
    cosine_similarity
)


class RecognitionService:

    FACE_THRESHOLD = 0.70
    VOICE_THRESHOLD = 0.75

    @staticmethod
    def compare_embeddings(
        embedding1,
        embedding2
    ):

        similarity = (
            cosine_similarity(
                [embedding1],
                [embedding2]
            )[0][0]
        )

        return float(similarity)

    @staticmethod
    def is_face_match(
        embedding1,
        embedding2
    ):

        similarity = (
            RecognitionService
            .compare_embeddings(
                embedding1,
                embedding2
            )
        )

        return {
            "matched": (
                similarity >=
                RecognitionService
                .FACE_THRESHOLD
            ),
            "similarity": similarity
        }

    @staticmethod
    def is_voice_match(
        embedding1,
        embedding2
    ):

        similarity = (
            RecognitionService
            .compare_embeddings(
                embedding1,
                embedding2
            )
        )

        return {
            "matched": (
                similarity >=
                RecognitionService
                .VOICE_THRESHOLD
            ),
            "similarity": similarity
        }