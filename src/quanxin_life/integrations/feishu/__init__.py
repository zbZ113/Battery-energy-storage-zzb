"""Safe, dependency-injected building blocks for Feishu event integration."""

from importlib import import_module
from typing import TYPE_CHECKING

from .analysis_plots import (
    ANALYSIS_PLOT_DISPATCHER_VERSION,
    AnalysisPlotPlan,
    AnalysisPlotTemplate,
    FeishuAnalysisPlotArtifact,
    FeishuAnalysisPlotError,
    FeishuAnalysisPlotter,
)
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
from .scenario_authorization import (
    AuditedScenarioResultAuthorizer,
    BlastScenarioResultAuthorizer,
    ScenarioAwareAuditedResultAuthorizer,
)
from .scenario_contexts import (
    FeishuScenarioContextRecord,
    ScenarioAnalysisInput,
    SqlAlchemyFeishuScenarioContextStore,
)
from .scenario_plot import (
    SCENARIO_PLOT_VERSION,
    FeishuScenarioPlotArtifact,
    FeishuScenarioPlotError,
    FeishuScenarioPlotter,
)
from .scenario_reports import (
    FeishuScenarioReportJob,
    FeishuScenarioReportResultFactory,
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

if TYPE_CHECKING:
    from .aily_scenarios import (
        AilyScenarioReferenceUseAuthorizer,
        RejectingAilyScenarioReferenceUseAuthorizer,
        SqlAlchemyAilyScenarioContextGateway,
    )
    from .aily_tasks import (
        AilyAnalysisJobDelivery,
        AilyBitableWriter,
        SqlAlchemyAilyAnalysisTaskGateway,
    )

_AILY_SCENARIO_EXPORTS = frozenset(
    {
        "AilyScenarioReferenceUseAuthorizer",
        "RejectingAilyScenarioReferenceUseAuthorizer",
        "SqlAlchemyAilyScenarioContextGateway",
    }
)
_AILY_TASK_EXPORTS = frozenset(
    {
        "AilyAnalysisJobDelivery",
        "AilyBitableWriter",
        "SqlAlchemyAilyAnalysisTaskGateway",
    }
)


def __getattr__(name: str) -> object:
    if name in _AILY_TASK_EXPORTS:
        module = import_module(".aily_tasks", __name__)
        return getattr(module, name)
    if name in _AILY_SCENARIO_EXPORTS:
        module = import_module(".aily_scenarios", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ANALYSIS_PLOT_DISPATCHER_VERSION",
    "BITABLE_RUN_FIELD_NAMES",
    "SCENARIO_PLOT_VERSION",
    "AilyAnalysisJobDelivery",
    "AilyBitableWriter",
    "AilyScenarioReferenceUseAuthorizer",
    "AnalysisPlotPlan",
    "AnalysisPlotTemplate",
    "AuditedCardBuilder",
    "AuditedCardError",
    "AuditedResultAuthorization",
    "AuditedScenarioResultAuthorizer",
    "BitableConflictError",
    "BitableProtocolError",
    "BitableValidationError",
    "BitableWriteAction",
    "BitableWriteResult",
    "BitableWriterError",
    "BlastScenarioResultAuthorizer",
    "FeishuAesCbcDecryptor",
    "FeishuAnalysisPlotArtifact",
    "FeishuAnalysisPlotError",
    "FeishuAnalysisPlotter",
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
    "FeishuScenarioContextRecord",
    "FeishuScenarioPlotArtifact",
    "FeishuScenarioPlotError",
    "FeishuScenarioPlotter",
    "FeishuScenarioReportJob",
    "FeishuScenarioReportResultFactory",
    "FeishuSignatureError",
    "FeishuWebhookSecrets",
    "FeishuWebhookVerifier",
    "RejectingAilyScenarioReferenceUseAuthorizer",
    "ScenarioAnalysisInput",
    "ScenarioAwareAuditedResultAuthorizer",
    "SqlAlchemyAilyAnalysisTaskGateway",
    "SqlAlchemyAilyScenarioContextGateway",
    "SqlAlchemyFeishuReceiptStore",
    "SqlAlchemyFeishuScenarioContextStore",
    "build_run_reference_card",
    "build_status_card",
    "parse_feishu_event_reference",
]
