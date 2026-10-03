from pymongo import AsyncMongoClient

from controllers.rag_controller import RAGController
from helpers.config import Setting
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.llm.llm_factory import ModelFactory
from services.llm.llm_manager import ModelManager
from services.processing.processing_service import ProcessingService
from services.rag.indexing_service import IndexingService
from services.rag.rag_service import RAGService
from services.rag.retrieval_service import RetrievalService
from services.storage.project_storage import ProjectStorage
from services.storage.upload_service import UploadService
from services.vectordb.vectordb_factory import VectorDBFactory
from services.vectordb.vectordb_manager import VectorDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ApplicationContainer:
    def __init__(
        self,
        settings: Setting,
    ):
        self.settings = settings

        self.mongo_conn: AsyncMongoClient | None = None
        self.db_client = None

        self.project_model: ProjectModel | None = None
        self.asset_model: AssetModel | None = None
        self.chunk_model: ChunkModel | None = None

        self.model_manager: ModelManager | None = None
        self.vector_db_manager: VectorDBManager | None = None

        self.project_storage: ProjectStorage | None = None
        self.upload_service: UploadService | None = None

        self.retrieval_service: RetrievalService | None = None
        self.indexing_service: IndexingService | None = None
        self.processing_service: ProcessingService | None = None

        self.rag_service: RAGService | None = None
        self.rag_controller: RAGController | None = None

    async def initialize(self) -> None:
        logger.info(
            "Initializing application dependencies",
        )

        self._initialize_storage()

        await self._initialize_mongodb()
        await self._initialize_models()
        await self._initialize_vector_db()
        await self._initialize_data_models()

        self._initialize_upload()
        self._initialize_processing()
        self._initialize_rag()

        logger.info(
            "Application dependencies initialized",
        )

    def _initialize_storage(self) -> None:
        self.project_storage = ProjectStorage()

        logger.info(
            "Project storage initialized",
        )

    async def _initialize_mongodb(self) -> None:
        database_config = self.settings.database

        self.mongo_conn = AsyncMongoClient(
            database_config.mongodb_url,
            timeoutMS=database_config.timeout_ms,
            connectTimeoutMS=database_config.connect_timeout_ms,
            maxPoolSize=database_config.max_pool_size,
            minPoolSize=database_config.min_pool_size,
            waitQueueTimeoutMS=database_config.wait_queue_timeout_ms,
        )

        self.db_client = self.mongo_conn[database_config.mongodb_database]

        await self.mongo_conn.admin.command(
            "ping",
        )

        logger.info(
            "MongoDB initialized: database=%s | max_pool=%s | min_pool=%s",
            database_config.mongodb_database,
            database_config.max_pool_size,
            database_config.min_pool_size,
        )

    async def _initialize_models(self) -> None:
        model_factory = ModelFactory(
            self.settings,
        )

        self.model_manager = ModelManager(
            model_factory,
        )

        self.model_manager.load_models()

        await self.model_manager.health_check_models()

        if self.settings.MODEL_WARMUP_ENABLED:
            await self.model_manager.warm_up_models(
                timeout_seconds=(self.settings.MODEL_WARMUP_TIMEOUT_SECONDS),
            )

        logger.info(
            "LLM models initialized",
        )

    async def _initialize_vector_db(self) -> None:
        vector_db_factory = VectorDBFactory(
            self.settings,
        )

        self.vector_db_manager = VectorDBManager(
            vector_db_factory,
        )

        self.vector_db_manager.load_databases()

        vector_db = self.vector_db_manager.get_database(
            "default",
        )

        await vector_db.create_collection()

        logger.info(
            "Vector database initialized",
        )

    async def _initialize_data_models(self) -> None:
        if self.db_client is None:
            raise RuntimeError(
                "Database client is not initialized",
            )

        self.project_model = ProjectModel(
            db_client=self.db_client,
        )

        await self.project_model.init_collection()

        self.asset_model = AssetModel(
            db_client=self.db_client,
        )

        await self.asset_model.init_collection()

        self.chunk_model = ChunkModel(
            db_client=self.db_client,
        )

        await self.chunk_model.init_collection()

        logger.info(
            "MongoDB data models initialized",
        )

    def _initialize_upload(self) -> None:
        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.asset_model is None:
            raise RuntimeError(
                "Asset model is not initialized",
            )

        if self.project_storage is None:
            raise RuntimeError(
                "Project storage is not initialized",
            )

        self.upload_service = UploadService(
            project_model=self.project_model,
            asset_model=self.asset_model,
            project_storage=self.project_storage,
            chunk_size=self.settings.files.default_chunk_size,
            max_file_size_mb=self.settings.files.max_size_mb,
            allowed_file_types=self.settings.files.allowed_types,
        )

        logger.info(
            "Upload service initialized",
        )

    def _initialize_processing(self) -> None:
        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.asset_model is None:
            raise RuntimeError(
                "Asset model is not initialized",
            )

        if self.chunk_model is None:
            raise RuntimeError(
                "Chunk model is not initialized",
            )

        if self.project_storage is None:
            raise RuntimeError(
                "Project storage is not initialized",
            )

        self.processing_service = ProcessingService(
            project_model=self.project_model,
            asset_model=self.asset_model,
            chunk_model=self.chunk_model,
            project_storage=self.project_storage,
        )

        logger.info(
            "Processing service initialized",
        )

    def _initialize_rag(self) -> None:
        if self.model_manager is None:
            raise RuntimeError(
                "Model manager is not initialized",
            )

        if self.vector_db_manager is None:
            raise RuntimeError(
                "Vector database manager is not initialized",
            )

        if self.project_model is None:
            raise RuntimeError(
                "Project model is not initialized",
            )

        if self.chunk_model is None:
            raise RuntimeError(
                "Chunk model is not initialized",
            )

        embedding_model = self.model_manager.get_selected_embedding_model()

        chat_model = self.model_manager.get_selected_chat_model()

        vector_db = self.vector_db_manager.get_database(
            "default",
        )

        self.retrieval_service = RetrievalService(
            embedding_model=embedding_model,
            vector_db=vector_db,
        )

        self.indexing_service = IndexingService(
            embedding_model=embedding_model,
            vector_db=vector_db,
            project_model=self.project_model,
            chunk_model=self.chunk_model,
        )

        self.rag_service = RAGService(
            chat_model=chat_model,
            retrieval_service=self.retrieval_service,
            indexing_service=self.indexing_service,
        )

        self.rag_controller = RAGController(
            rag_service=self.rag_service,
        )

        logger.info(
            "RAG dependencies initialized",
        )

    async def close(self) -> None:
        logger.info(
            "Closing application dependencies",
        )

        if self.vector_db_manager is not None:
            await self.vector_db_manager.close()

        if self.model_manager is not None:
            await self.model_manager.close()

        if self.mongo_conn is not None:
            await self.mongo_conn.close()

        logger.info(
            "Application dependencies closed",
        )
