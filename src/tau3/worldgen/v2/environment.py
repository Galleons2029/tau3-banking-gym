"""Tau environment adapter for V2 worlds, including hidden outcome assertions."""

import json
import re
from pathlib import Path

from tau3.environment.environment import Environment
from tau3.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau3.worldgen.v2.pipeline import check_certificate
from tau3.worldgen.v2.runtime import OperationRuntime, WorldDB, digest, goal_errors
from tau3.worldgen.v2.specs import WorldSpec


def load_spec(root: Path, allow_draft: bool = False) -> WorldSpec:
    """Load published certificates; draft loading is explicit and development-only."""
    manifest = json.loads((root / "manifest.json").read_text())
    if not allow_draft:
        if manifest.get("status") != "published":
            raise ValueError("World has not passed publication gates")
        certificate = check_certificate(root)
        if manifest.get("certificate_hash") != digest(certificate):
            raise ValueError("Published certificate identity changed")
        if manifest.get("purpose") == "paper_reproduction":
            from tau3.worldgen.v2.behaviour import check_qualification

            if manifest.get("behaviour_hash") != digest(check_qualification(root)):
                raise ValueError("Published behavioural certificate changed")
    return WorldSpec.model_validate_json((root / "spec.json").read_text())


class WorldTools(ToolKitBase):
    """Only public read/discovery routes are exposed; specifications stay private."""

    def __init__(self, spec, db, initial, retrieval_tools=None, public_documents=None):
        super().__init__(db)
        self.spec, self.initial = spec, initial
        self.retrieval_tools = retrieval_tools
        self.public_documents = public_documents
        self.unlocked: set[str] = set()
        self.user_grants: set[str] = set()

    @property
    def tools(self):
        exposed = dict(super().tools)
        if self.public_documents is None or not self.spec.public_document_catalog:
            exposed.pop("KB_list_documents", None)
            exposed.pop("KB_read_documents", None)
        if not self.spec.public_record_snapshot:
            exposed.pop("get_customer_records", None)
        if self.retrieval_tools:
            exposed.update(
                {
                    name: method
                    for name, method in self.retrieval_tools.tools.items()
                    if name
                    in {
                        "KB_search",
                        "KB_search_bm25",
                        "KB_search_dense",
                        "grep",
                        "shell",
                    }
                }
            )
        return exposed

    @is_tool(ToolType.READ)
    def KB_list_documents(self, query: str = "", offset: int = 0) -> str:
        """List public document IDs and titles; use category or product words.

        All query tokens must occur in the document ID or title. Empty query lists
        the entire public catalog in stable pages of 100. Use KB_read_documents to
        read selected IDs, including terms and servicing rules, without re-searching.

        Args:
            query: Words from the customer request or already retrieved public text.
            offset: Zero-based page offset; use next_offset to continue.
        """
        if self.public_documents is None or offset < 0:
            raise ValueError("Public catalog unavailable or invalid offset")
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        docs = [
            {"id": d.id, "title": d.title}
            for _, d in sorted(self.public_documents.items())
            if tokens <= set(re.findall(r"[a-z0-9]+", (d.id + " " + d.title).lower()))
        ]
        return json.dumps(
            {
                "documents": docs[offset : offset + 100],
                "total": len(docs),
                "next_offset": offset + 100 if offset + 100 < len(docs) else None,
            }
        )

    @is_tool(ToolType.READ)
    def KB_read_documents(self, document_ids: list[str]) -> str:
        """Read up to ten complete public documents by their exact catalog/search IDs.

        Args:
            document_ids: Exact public document IDs returned by catalog or search.
        """
        if self.public_documents is None or not 1 <= len(document_ids) <= 10:
            raise ValueError("Provide between one and ten public document IDs")
        if len(set(document_ids)) != len(document_ids) or any(
            k not in self.public_documents for k in document_ids
        ):
            raise ValueError("Duplicate or unknown public document ID")
        return json.dumps([self.public_documents[k].model_dump() for k in document_ids])

    def _runtime(self):
        return OperationRuntime(self.spec, self.db)

    def _contract(self, capability, actor):
        runtime = self._runtime()
        if capability not in runtime.aliases:
            raise ValueError("Unknown discoverable tool")
        _, operation = runtime.operations[runtime.aliases[capability]]
        if operation.actor != actor:
            raise ValueError("Wrong tool owner")
        return operation

    @is_tool(ToolType.READ)
    def get_current_time(self) -> str:
        """Return the world's current date; use it for time-dependent policies."""
        return self.spec.clock

    @is_tool(ToolType.READ)
    def find_customer(self, email: str) -> str:
        """Find a customer by their email.

        Args:
            email: Email supplied by the customer.
        """
        return json.dumps(
            [
                {"user_id": uid, **u}
                for uid, u in self.db.users.items()
                if u["email"] == email
            ]
        )

    @is_tool(ToolType.READ)
    def get_customer_records(self, user_id: str) -> str:
        """Read the identified customer's records, including their record IDs.

        Args:
            user_id: Customer identifier obtained from the user or lookup.
        """
        if user_id not in self.db.users:
            raise ValueError("Unknown customer")
        return json.dumps(
            {
                table: {
                    rid: row for rid, row in rows.items() if row["user_id"] == user_id
                }
                for table, rows in self.db.tables.items()
            }
        )

    def public_customer_records(self, user_id: str) -> dict:
        """Reconstruct public records through documented queries for blind audits.

        This is not an exposed agent tool. It never reads private goals or raw
        rows; each item comes from a normal scoped query. Missing projections
        remain missing rather than being completed from the private database.
        """
        if self.spec.public_record_snapshot:
            return json.loads(self.get_customer_records(user_id))
        import itertools

        result, known_ids, visited = {}, set(), set()
        runtime = self._runtime()
        for _ in range(8):
            before = len(visited)
            for category in self.spec.categories:
                for op in category.operations:
                    if op.kind != "read":
                        continue
                    params = list(op.query.filters.values())
                    combinations = (
                        itertools.product(sorted(known_ids), repeat=len(params))
                        if params
                        else [()]
                    )
                    for values in combinations:
                        for product in category.products:
                            args = {
                                "user_id": user_id,
                                "product_id": product.id,
                                "offset": 0,
                                **dict(zip(params, values)),
                            }
                            key = digest([op.id, args])
                            if key in visited:
                                continue
                            visited.add(key)
                            if len(visited) > 10000:
                                raise ValueError(
                                    "Public query traversal exceeds audit bound"
                                )
                            alias = next(
                                a for a, oid in runtime.aliases.items() if oid == op.id
                            )
                            self.unlock_discoverable_agent_tool(alias)
                            while True:
                                reply = json.loads(
                                    self.call_discoverable_agent_tool(
                                        alias, json.dumps(args)
                                    )
                                )
                                for row in reply["records"]:
                                    rid = row["record_id"]
                                    known_ids.add(rid)
                                    result.setdefault(op.query.table, {})[rid] = {
                                        k: v for k, v in row.items() if k != "record_id"
                                    }
                                if reply["next_offset"] is None:
                                    break
                                args["offset"] = reply["next_offset"]
            if len(visited) == before:
                break
        return result

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def unlock_discoverable_agent_tool(self, capability: str) -> str:
        """Unlock an exact tool name found in the knowledge base.

        Args:
            capability: Exact documented tool name including its suffix.
        """
        operation = self._contract(capability, "assistant")
        self.unlocked.add(capability)
        return json.dumps(
            {
                "name": capability,
                "description": operation.description,
                "parameters": {
                    k: v.model_dump() for k, v in operation.parameters.items()
                },
            }
        )

    @is_tool(ToolType.WRITE)
    def call_discoverable_agent_tool(self, capability: str, arguments: str) -> str:
        """Call an unlocked operation with a JSON object of arguments.

        Args:
            capability: Exact unlocked name.
            arguments: JSON object containing every required argument.
        """
        if capability not in self.unlocked:
            raise ValueError("Tool is not unlocked")
        return json.dumps(self._runtime().execute(capability, json.loads(arguments)))

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def give_discoverable_user_tool(self, capability: str) -> str:
        """Give the customer a documented tool they must execute themselves.

        Args:
            capability: Exact user-owned tool name from the knowledge base.
        """
        operation = self._contract(capability, "user")
        self.user_grants.add(capability)
        return json.dumps(
            {
                "name": capability,
                "description": operation.description,
                "parameters": {
                    k: v.model_dump() for k, v in operation.parameters.items()
                },
            }
        )

    def check_scenario_outcome(self, scenario_id: str) -> bool:
        """Evaluator-only assertion; deliberately not exposed as an agent tool."""
        scenario = next(
            (
                s
                for c in self.spec.categories
                for s in c.scenarios
                if s.id == scenario_id
            ),
            None,
        )
        if scenario is None:
            return False
        return not goal_errors(
            self.initial, self.db, scenario.goals, scenario.user_id, self.spec
        )


class WorldUserTools(ToolKitBase):
    """User-side routes share business state but cannot call assistant tools."""

    def __init__(self, agent):
        super().__init__(agent.db)
        self.agent = agent

    @is_tool(ToolType.READ)
    def list_granted_tools(self) -> str:
        """List the exact names and argument schemas of tools granted to you.

        Call this when the assistant asks you to perform an action. Use the
        returned name as capability in call_discoverable_user_tool; a product
        identifier or description is not a capability. No ungranted tool is shown.
        """
        return json.dumps(
            [
                {
                    "name": capability,
                    "description": operation.description,
                    "parameters": {
                        k: v.model_dump() for k, v in operation.parameters.items()
                    },
                }
                for capability in sorted(self.agent.user_grants)
                for operation in [self.agent._contract(capability, "user")]
            ]
        )

    @is_tool(ToolType.WRITE)
    def call_discoverable_user_tool(self, capability: str, arguments: str) -> str:
        """Execute a tool that the assistant gave you; list_granted_tools shows names.

        Args:
            capability: Exact granted tool name.
            arguments: Required tool arguments encoded as a JSON object.
        """
        if capability not in self.agent.user_grants:
            raise ValueError("User tool was not granted")
        return json.dumps(
            OperationRuntime(self.agent.spec, self.db).execute(
                capability, json.loads(arguments), "user"
            )
        )


class WorldEnvironment(Environment):
    """Ensure both actors retain a single isolated database during replay."""

    def sync_tools(self):
        if self.tools is not None and self.user_tools is not None:
            self.user_tools.db = self.tools.db


def get_environment(
    root: Path,
    db=None,
    retrieval_variant=None,
    retrieval_kwargs=None,
    task=None,
    solo_mode=False,
    **kwargs,
):
    """Construct a V2 Tau environment with the established retrieval pipelines."""
    from tau3.domains.banking_knowledge.data_model import KnowledgeBase, TransactionalDB
    from tau3.domains.banking_knowledge.retrieval import build_tools, resolve_variant
    from tau3.worldgen.world import allow_draft

    if solo_mode:
        raise ValueError("banking_synth does not support solo mode")
    spec = load_spec(root, allow_draft())
    initial = WorldDB.model_validate_json((root / "db.json").read_text())
    state = db or initial.model_copy(deep=True)
    knowledge = KnowledgeBase.load(str(root / "documents"))
    variant_name = retrieval_variant or "bm25_grep"
    inline = ""
    helper = None
    if variant_name in {"golden_retrieval", "full_kb"}:
        selected = (
            (set(task.required_documents or []) if task else set())
            if variant_name == "golden_retrieval"
            else set(knowledge.documents)
        )
        inline = "\n\n".join(
            f"# {d.title}\n{d.content}"
            for k, d in knowledge.documents.items()
            if k in selected
        )
    elif variant_name != "no_knowledge":
        variant = resolve_variant(variant_name, **(retrieval_kwargs or {}))
        # Only retrieval methods are exposed by WorldTools; no legacy business
        # method or legacy database record becomes visible to either actor.
        helper = build_tools(variant, TransactionalDB(), knowledge)
    agent = WorldTools(
        spec, state, initial, helper, knowledge.documents if helper else None
    )
    user = WorldUserTools(agent)
    policy = (
        "You provide support for the fictional Rho-Bank. Use the knowledge base for product terms, procedures and tool names. Identify the customer before reading their records. Ask the customer for missing customer-specific identifiers, including their requested identifier for a newly created record. Never invent policies, rates or operations. Obtain current dates from get_current_time. Discover tools by exact documented names and unlock before calling. User-owned tools must be given to the user to execute; do not execute them yourself. After granting, tell the user the exact capability name including its suffix and all required arguments. The user can query list_granted_tools to inspect their grants. If documentation is insufficient, say so. An operation is complete only after the tool succeeds. Explain eligibility refusals accurately. When a public document catalog is available, list titles for the requested category or product and batch-read its terms, procedures and servicing rules. Keep already retrieved facts; do not repeatedly search for the same fact or an identifier the customer must supply. After all requested operations succeed, give one concise final summary so the customer can finish the conversation.\n\n"
        + inline
    )
    return WorldEnvironment("banking_synth", policy, agent, user)
