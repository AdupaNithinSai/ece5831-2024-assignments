from pricing import get_net_price
from product import get_tax as product_get_tax
import sys
import billing
from sales import TAX_RATE, create_sales_order, create_delivery

net_price = get_net_price(price=100, tax_rate=0.01)
print(f"Net Price: {net_price}")

tax = product_get_tax(100)
print(f"Tax from product module: {tax}")


def display_search_path():
    print("Current Module Search Path:")
    for path in sys.path:
        print(path)

display_search_path()


billing.print_billing_doc()


def main():
    print(f"Default Sales Tax Rate: {TAX_RATE}")
    create_sales_order()
    create_delivery()

if __name__ == "__main__":
    main()


from sales.order import create_sales_order as create_order
from sales.delivery import create_delivery as start_delivery
from sales.billing import create_billing as issue_billing


create_order()
start_delivery()
issue_billing()