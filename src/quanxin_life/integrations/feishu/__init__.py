"""Safe, dependency-injected building blocks for Feishu event integration."""

from .bitable import (
    BITABLE_RUN_FIELD_NAMES,
    BitableConflictError,
    BitableProtocolError,
    BitableValidationError,
    BitableWriteAction,
    BitableWriterError,
    BitableWriteResult,
    FeishuBitableWriter,
)
from .cards import (
    AuditedCardBuilder,
    AuditedCardError,
    AuditedResultAuthorization,
    FeishuCardStatus,
    build_run_reference_card,
    build_status_card,
)
from .decryptor import FeishuAesCbcDecryptor, FeishuDecryptorError
from .events import (
    FeishuEventError,
    FeishuEventOutcome,
    FeishuEventProcessor,
    FeishuPayloadDecryptionUnavailable,
    FeishuPayloadDecryptor,
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
    FeishuReceiptStore,
)
from .report_delivery import (
    FeishuReportDelivery,
    FeishuReportDeliveryError,
    FeishuReportDeliveryReceipt,
)
from .routing import (
    FeishuEventReference,
    FeishuEventReferenceError,
    FeishuInboundEventKind,
    parse_feishu_event_reference,
)
from .security import (
    FeishuSignatureError,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)
from .sqlalchemy_receipts import (
    FeishuReceiptOwnershipError,
    SqlAlchemyFeishuReceiptStore,
)

__all__ = [
    "BITABLE_RUN_FIELD_NAMES",
    "AuditedCardBuilder",
    "AuditedCardError",
    "AuditedResultAuthorization",
    "BitableConflictError",
    "BitableProtocolError",
    "BitableValidationError",
    "BitableWriteAction",
    "BitableWriteResult",
    "BitableWriterError",
    "FeishuAesCbcDecryptor",
    "FeishuBitableWriter",
    "FeishuCardStatus",
    "FeishuDecryptorError",
    "FeishuEventError",
    "FeishuEventOutcome",
    "FeishuEventProcessor",
    "FeishuEventReference",
    "FeishuEventReferenceError",
    "FeishuInboundEventKind",
    "FeishuPayloadDecryptionUnavailable",
    "FeishuPayloadDecryptor",
    "FeishuReceiptClaim",
    "FeishuReceiptClaimStatus",
    "FeishuReceiptOwnershipError",
    "FeishuReceiptStore",
    "FeishuReportDelivery",
    "FeishuReportDeliveryError",
    "FeishuReportDeliveryReceipt",
    "FeishuSignatureError",
    "FeishuWebhookSecrets",
    "FeishuWebhookVerifier",
    "SqlAlchemyFeishuReceiptStore",
    "build_run_reference_card",
    "build_status_card",
    "parse_feishu_event_reference",
]
