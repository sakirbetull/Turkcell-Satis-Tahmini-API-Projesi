
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL bağlantı bilgilerini doldur
db_user = "postgres"
db_password = "12345"
db_host = "localhost"  # veya sunucu adresi
db_port = "5432"  # PostgreSQL varsayılan portu
db_name = "gyk"

# SQLAlchemy bağlantı motorunu oluştur
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Tabloyu pandas DataFrame olarak çek
table_name = "tabledf"
df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

# Çekilen veriyi görüntüle
print(df.head())

df.to_csv("tabledf.csv", index=False)




'''from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL bağlantı bilgileri
DB_USERNAME = "postgres"
DB_PASSWORD = "12345"
DB_HOST = "localhost"  # Eğer uzak sunucuysa IP adresini yazın
DB_PORT = "5432"
DB_NAME = "gyk"

# PostgreSQL bağlantı dizesi (SQLAlchemy için)
db_url = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy engine oluştur
engine = create_engine(db_url)

query = """

create view tabledf as
select
	o.order_date, --dönem bazlı
	c.customer_id,
	c.city,
	p.product_id,
	p.product_name,
	p.units_in_stock,
	od.unit_price,
	od.quantity,
	od.discount,
	ca.category_id,
	ca.category_name

	from orders o
inner join order_details od on o.order_id = od.order_id
inner join products p on od.product_id = p.product_id
inner join customers c on o.customer_id = c.customer_id
inner join categories ca on p.category_id = ca.category_id;
"""

# Sorguyu çalıştır ve DataFrame olarak al
df = pd.read_sql(query, engine)

# Sonuçları göster
print(df.head())

# CSV olarak kaydet (VS Code'da açmak için)
df.to_csv("sales_data.csv", index=False)
'''