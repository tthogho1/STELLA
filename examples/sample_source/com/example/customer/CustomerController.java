package com.example.customer;

class CustomerController {

    public CustomerResponse create(CustomerRequest request) {
        return customerService.register(request);
    }
}
