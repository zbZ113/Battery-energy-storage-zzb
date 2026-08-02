export type UserRole = "ADMIN" | "MEMBER" | "JUDGE";

export interface AuthPrincipal {
  user_id: string;
  username: string;
  role: UserRole;
  must_change_password: boolean;
}

export interface ProjectRecord {
  project_id: string;
  name: string;
  owner_user_id: string;
  created_at: string;
}

export interface AgentRunRecord {
  run_id: string;
  project_id: string;
  status: string;
  plan_id?: string | null;
  pending_approval_id?: string | null;
  created_at: string;
  updated_at: string;
  [key: string]: unknown;
}

export interface AgentRunCreateRequest {
  project_id: string;
  user_goal: string;
  dataset_ids: string[];
  requested_outputs: string[];
}

export interface AgentEvent {
  event_id: string;
  run_id: string;
  sequence: number;
  event_type: string;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface ToolResult {
  result_id: string;
  tool_name: string;
  tool_version: string;
  input_hash: string;
  values: Record<string, unknown>;
  uncertainty: Record<string, unknown> | null;
  provenance: ProvenanceRecord[];
  warnings: string[];
  model_version?: string | null;
  data_version?: string | null;
  feature_version?: string | null;
  created_at: string;
}

export interface ProvenanceRecord {
  source_id: string;
  source_kind: string;
  uri: string;
  sha256: string;
  description: string;
  created_at: string;
}
