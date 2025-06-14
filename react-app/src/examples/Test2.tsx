import { useState } from "react";

const contacts = [
    { id: 0, name: 'Taylor', email: 'taylor@mail.com' },
    { id: 1, name: 'Alice', email: 'alice@mail.com' },
    { id: 2, name: 'Bob', email: 'bob@mail.com' }
  ];

function ContactList({
    selectedContact,
    contacts,
    onSelect
}: any){
    return (
        <section className="contact-list">
            <ul>
                {
                    contacts.map((contact: any) => 
                        <li key={contact.id}>
                            <button onClick={()=>{
                                onSelect(contact);
                            }}>
                                {contact.name}
                            </button>
                        </li>
                    )
                }
            </ul>
        </section>
    )
}

function Chat({contact}: any){
    const [text, setText] = useState('');
    return (
        <section>
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={`Message ${contact.name}`} />
        </section>
    )
}
export default function Msg(){
    const [to, setTo] = useState(contacts[0]);

    return (
        <div>
            <ContactList contacts={contacts} selectedContact={to} onSelect={contact => setTo(contact)} />
            <Chat key={to.id} contact={to} />
        </div>
    )
}