class MemoryService:

    memory_store = {}

    @staticmethod
    def save_memory(user_id, text):

        if user_id not in (
            MemoryService.memory_store
        ):

            MemoryService.memory_store[
                user_id
            ] = []

        MemoryService.memory_store[
            user_id
        ].append(text)

    @staticmethod
    def get_memories(user_id):

        return (
            MemoryService.memory_store
            .get(user_id, [])
        )