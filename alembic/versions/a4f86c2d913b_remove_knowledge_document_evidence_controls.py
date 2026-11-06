"""remove evidence controls from knowledge documents

Revision ID: a4f86c2d913b
Revises: c1e7a9b4d2f6
Create Date: 2026-11-05 16:11:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4f86c2d913b"
down_revision: Union[str, Sequence[str], None] = "c1e7a9b4d2f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_knowledge_documents_synthetic_not_public",
        "knowledge_documents",
        type_="check",
    )
    op.drop_constraint(
        "ck_knowledge_documents_evidence_status",
        "knowledge_documents",
        type_="check",
    )
    op.drop_column("knowledge_documents", "last_reviewed_at")
    op.drop_column("knowledge_documents", "expiry_date")
    op.drop_column("knowledge_documents", "effective_date")
    op.drop_column("knowledge_documents", "usable_for_public_answers")
    op.drop_column("knowledge_documents", "usable_for_demo")
    op.drop_column("knowledge_documents", "is_verified")
    op.drop_column("knowledge_documents", "evidence_status")


def downgrade() -> None:
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "evidence_status",
            sa.String(length=20),
            server_default=sa.text("'unverified'"),
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "is_verified",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "usable_for_demo",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "usable_for_public_answers",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column("expiry_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_knowledge_documents_evidence_status",
        "knowledge_documents",
        "evidence_status IN ('verified', 'historical', 'mixed', 'unverified', 'synthetic')",
    )
    op.create_check_constraint(
        "ck_knowledge_documents_synthetic_not_public",
        "knowledge_documents",
        "evidence_status != 'synthetic' OR usable_for_public_answers = false",
    )
