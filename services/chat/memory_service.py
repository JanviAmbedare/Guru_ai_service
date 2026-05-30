class MemoryService:

    memory_store = {}

    MAX_MEMORY = 20

    @staticmethod
    def save_memory(
        user_id,
        text
    ):

        if user_id not in (
            MemoryService.memory_store
        ):

            (
                MemoryService
                .memory_store[user_id]
            ) = []

        (
            MemoryService
            .memory_store[user_id]
            .append(text)
        )

        if len(
            MemoryService
            .memory_store[user_id]
        ) > MemoryService.MAX_MEMORY:

            (
                MemoryService
                .memory_store[user_id]
                .pop(0)
            )

    @staticmethod
    def get_memories(user_id):

        return (
            MemoryService
            .memory_store
            .get(user_id, [])
        )

    @staticmethod
    def clear_memory(user_id):

        if user_id in (
            MemoryService.memory_store
        ):

            del (
                MemoryService
                .memory_store[user_id]
            )