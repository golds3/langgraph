CREATE TABLE ai.sign_inf (
	Id BIGINT auto_increment NOT NULL,
	acct_no varchar(100) NOT NULL COMMENT '签约账号',
	reg_type varchar(100) NOT NULL COMMENT '监管类型 10-存款监管 01-票据监管 11-存款+票据监管',
	email_flag varchar(100) DEFAULT 0 NOT NULL COMMENT '客户邮件通知功能 0-不开通 1-开通,如果为1，则必须要同步维护ai.email_inf表数据',
	bill_reg_type varchar(100) NULL COMMENT '(如果监管类型包含票据，这个字段一定要维护)票据监管类型 100-质押 110-质押+背书 111-质押+背书+贴现 101-质押+贴现 011-背书+贴现',
	CONSTRAINT sign_inf_pk PRIMARY KEY (Id),
	CONSTRAINT sign_inf_unique UNIQUE KEY (acct_no)
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
CREATE INDEX sign_inf_acct_no_IDX USING BTREE ON ai.sign_inf (acct_no);



CREATE TABLE ai.email_inf (
	Id BIGINT auto_increment NOT NULL,
	acct_no varchar(100) NOT NULL COMMENT '签约账号',
	email_address varchar(100) NOT NULL COMMENT '邮箱地址',
	CONSTRAINT email_inf_pk PRIMARY KEY (Id),
	CONSTRAINT email_inf_unique UNIQUE KEY (acct_no,email_address)
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;
CREATE INDEX email_inf_acct_no_IDX USING BTREE ON ai.email_inf (acct_no);





INSERT INTO ai.sign_inf (acct_no, reg_type, email_flag, bill_reg_type)
VALUES
('ACCT123456', '10', 1, '100'),
('ACCT789012', '01', 0, '110'),
('ACCT345678', '11', 1, '111'),
('ACCT901234', '10', 0, NULL),
('ACCT567890', '01', 1, '011');

INSERT INTO ai.email_inf (acct_no, email_address) VALUES
('ACCT123456', 'user1@example.com'),
('ACCT345678', 'user2@example.com'),
('ACCT567890', 'user3@example.com');
