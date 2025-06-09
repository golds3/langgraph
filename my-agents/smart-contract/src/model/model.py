from typing import Optional

from pydantic import BaseModel


class Contract(BaseModel):
    acct_no: str
    reg_type: str
    email_flag: int = 0
    bill_reg_type: Optional[str] = None

class EmailInf(BaseModel):
    acct_no: str
    email_address: str

