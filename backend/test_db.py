import asyncio
from kartikey.infrastructure.database import async_session_maker
from kartikey.analysis.models import Analysis
from sqlalchemy import select
import json

async def run():
    async with async_session_maker() as session:
        result = await session.execute(select(Analysis).order_by(Analysis.created_at.desc()).limit(1))
        analysis = result.scalar_one_or_none()
        if not analysis:
            return
        
        print(f"Analysis ID: {analysis.id}")
        is_numbers = []
        for f in analysis.findings:
            for s in (f.applicable_standards or []):
                base = getattr(s, 'base_designation', None) or getattr(s, 'is_number', None)
                if base and base not in is_numbers:
                    is_numbers.append(base)
        print(f"Base Designations: {is_numbers}")

asyncio.run(run())
