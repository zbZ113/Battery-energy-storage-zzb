"""Safe, dependency-injected building blocks for Feishu event integration."""

from .cards import build_run_reference_card
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
    "FeishuEventError",
    "FeishuEventOutcome",
    "FeishuEventProcessor",
    "FeishuPayloadDecryptionUnavailable",
    "FeishuPayloadDecryptor",
    "FeishuReceiptClaim",
    "FeishuReceiptClaimStatus",
    "FeishuReceiptOwnershipError",
    "FeishuReceiptStore",
    "FeishuSignatureError",
    "FeishuWebhookSecrets",
    "FeishuWebhookVerifier",
    "SqlAlchemyFeishuReceiptStore",
    "build_run_reference_card",
]
